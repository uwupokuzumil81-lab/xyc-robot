# ACT 和 π0.5 的训练、保存与推理

当前已完成模型侧的闭环：现有 LeRobot v3 数据 → 四路图像与18维状态 → ACT训练 → 平台checkpoint → 重载 → 50×18动作预测。
正式长训练尚未启动；测试权重仅验证计算流程。`learning.py deploy` 已通过公共运行时和回放控制后端接到同一套18维实机控制，
但还没有用有效训练权重完成实机策略验证；它当前是电脑/GPU进程经本机转发控制ROS后端，不是树莓派内的独立ROS策略节点。
云服务器推理传输也尚未实现，不能把离线测试权重当成已经能完成任务的模型。

## 哪些统一，哪些由模型决定

| 环节 | 平台公共代码 | ACT 专用代码 |
| --- | --- | --- |
| 读取数据 | learning/dataset.py：按声明读取v3索引、视频、任务文本和时间窗口 | `sample_spec()`声明state/图像历史、动作偏移与任务文本需求；适配器处理图像尺寸 |
| 统计与归一化 | DatasetMetadata按声明只计算训练回合的统计量 | `statistics_spec()`声明mean/std或分位数等；适配器执行归一化和反归一化 |
| 配置 | learning/config.py：路径、训练步数、优化器、保存频率 | models/act/configuration_act.py：ACT网络、VAE、chunk参数 |
| 输入输出 | contracts.py：相机名、状态/动作顺序、单位 | models/act/adapter.py：图像缩放与归一化、张量打包、反归一化 |
| 训练 | learning/trainer.py：优化、验证、日志、续训 | 适配器提供loss、参数组、验证指标与权重，网络代码在modeling_act.py |
| 保存恢复 | learning/checkpoint.py：权重、资产、配置、统计量、训练状态 | 无另一套LeRobot保存/恢复接口 |
| 推理 | runtime/：逐帧执行、策略选择、18维命令映射；learning/inference.py负责离线数据入口 | ACT输出动作chunk，可选时间集成 |
| 机器人驱动 | 原有ROS功能包 | 模型不能自行打开串口或绕过驱动 |

以后接模型，新增 `models/<name>/` 的配置、模型、适配器，在 `learning/registry.py` 注册并给一个训练YAML。
适配器提供 `compute_loss`、`predict_chunk`、`validation_spec`、`validation_metrics`、`optimizer_groups`、`reset`。
`validation_spec()`声明主指标及min/max方向；`validation_metrics()`返回每项独立的 `MetricValue(value, weight)`，公共训练器加权汇总，不读取某个模型的padding或指标名称。
`model_config()` 导出可恢复的模型构造配置；公共训练器不读取 ACT 的配置字段。
`training_options(options, smoke=..., no_pretrained=...)` 负责模型自己的短训和预训练开关。
构建模型之前，公共训练器先向注册的适配器类请求 `sample_spec(model_config, schema)` 和
`statistics_spec(model_config, schema)`，按其声明准备数据与统计量；图像常量也由适配器声明，公共数据层不再写死ImageNet参数。
`SampleSpec`支持当前帧、state/图像历史帧、任意动作偏移和`include_task`；窗口越过回合边界时返回填充标记，不跨回合取数据。
`StatisticsSpec`支持mean/std、min/max、q01/q99及模型声明的常量。文本按每帧task_index解析；分词和特殊编码属于模型适配器。
checkpoint记录采样声明、统计声明和训练集统计量。公共读取器保留原始RGB分辨率。ACT 和 π0.5 各自用同一个 prepare 路径处理训练、离线推理和实时图像。

运行时已拆成三层：

```text
PolicyRunner                 调模型并记录预测，交给选定的推理策略
  └─ strategies/
       ├─ sync_chunk.py      普通动作块队列；ACT可选时间集成
       └─ rtc.py             后台预测、动作接续、过期检查，用于提供RTC推理的模型
PolicyExecutor               把选中的18维动作映射成统一机器人命令并交给CommandSink
```

RTC 的公共调度层运行单个后台预测任务，在旧块尚有动作时提交下一次预测。切换时按实际已发送的动作步数
跳过新块的重叠前缀。新块返回时，如果接续位置已经越过引导区间，就丢弃这次预测。旧块还有有效动作时继续执行，并按实测延迟重新预测。π0.5 适配器用归一化动作做模型内的 RTC 引导去噪，并返回反归一化后的机器人动作。
ACT 使用 `sync_chunk`。两种策略都把单步动作交给 `PolicyExecutor` 和现有的机器人控制后端。

普通动作块策略通过`chunk_execution_spec()`读取执行步数、预测时域和是否集成；
需要集成时只调用适配器的`ensemble_action(chunk)`。公共运行时不再访问ACT内部的`policy.temporal_ensembler`。

ACT源码固定自 LeRobot v0.4.3；`models/act/PROVENANCE.json` 保存上游地址、原文件哈希和修改说明。
修改仅是本地配置、常量和基类依赖接线；网络、VAE和损失计算保留该版本实现。Apache-2.0许可证随模型保存。
π0.5 源码固定自 LeRobot v0.5.1。模型、预处理和 RTC 引导都放在 `models/pi05/`，
上游文件哈希和改动记录在 `PROVENANCE.json`，并附有 Apache-2.0 许可证。
ACT 和 π0.5 的运行环境都无需安装 LeRobot 包。π0.5 使用 PyTorch 和 Transformers，基座与 tokenizer 作为独立资产加载。

## π0.5 训练与推理

π0.5 直接读取同一份 LeRobot v3 数据集。四路 RGB 按 `forward, chest, wrist_left, wrist_right` 进入模型。
本地 π0.5 处理器把每帧任务文字和归一化后的 18 维 state 组装并分词。状态和关节动作使用训练集分位数，分位数重合时使用观测范围。
升降和底盘动作固定以 `[-1, 1]` 归一化，保留原有三值标签。动作专家内部宽度为 32，输出为 `50 × 18`，
回合末尾补帧及第 19–32 列不参与损失。执行阶段的关节限位、底盘三值映射及升降保护由公共运行时和控制器处理。

在 GPU 电脑上使用 Python 3.12 的独立环境。
在 Hugging Face 上接受 [PaliGemma tokenizer](https://huggingface.co/google/paligemma-3b-pt-224) 的访问条件，然后登录。训练和重载权重均使用此 tokenizer 的本机缓存。

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n xyc-pi05 python=3.12 pip -y
conda activate xyc-pi05
cd ~/Projects/blue_block_to_brown_bowl
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e 'code[pi05]'
hf auth login

python code/tools/learning.py train --config code/config/train/pi05.yaml \
  --dataset "$PWD/datasets/xyc/blue_block_to_brown_bowl_v1" \
  --output "$PWD/weights/pi05/run_001"
```

默认从 [LeRobot 的 π0.5 基座](https://huggingface.co/lerobot/pi05_base) 加载 PyTorch 权重，微调视觉编码器、VLM 和动作专家。
第一次加载需下载约 14.5 GB。训练参数在 `config/train/pi05.yaml`。这个配置以 batch 1、10000 步为起点，正式续训使用公共 `--resume`。
全参训练需要显存明显高于本机 8 GB。基座损坏、权重缺失或者张量形状不匹配时训练会报错。
π0.5 验证使用独立生成的固定去噪初始噪声。同一组验证样本的分数可用于比较各次 checkpoint，验证过程不推进训练随机数。

```bash
python code/tools/learning.py infer --checkpoint "$PWD/weights/pi05/run_001" \
  --dataset "$PWD/datasets/xyc/blue_block_to_brown_bowl_v1" \
  --episode 0 --device cuda --output /tmp/xyc-pi05-prediction.json

python code/tools/learning.py deploy --checkpoint "$PWD/weights/pi05/run_001" \
  --device cuda --prepare-mode keyboard --strategy rtc \
  --task "Pick up the blue rectangular block and place it into the brown bowl."
```

实机推理按下文的统一控制通道启动 ROS 后端。按 P 后先在保持姿态下等待首块。连续运行时，`n_action_steps`
是 RTC 的常规重规划间隔，默认 10 个控制步。剩余动作不足以覆盖估计延迟时会提前重规划。
后台预测报错、旧块耗尽且新块未就绪，或动作块超出自身预测时域时，走现有安全停车。每条命令
仍校验新鲜反馈和 250 ms 的有效期。RTC 使用现有 π0.5 checkpoint 的推理时引导，可用 `--strategy sync_chunk` 选择同步方式。

## 安装和启动

推荐在云GPU上单独创建环境，不要装进树莓派 `xyc`：

```bash
conda create -n xyc_act python=3.10 -y
conda activate xyc_act
cd /你的路径/pi05_from_zero
python -m pip install ".[act]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

依赖固定torch2.7.1、torchvision0.22.1和einops0.8.1。Windows、Linux使用相同Python入口。
如GPU供应商已提供匹配的torch/torchvision，先核对版本和CUDA可用性，再安装项目，避免在CPU版torch环境中误启动GPU训练。
命令行建议使用源码入口 `python tools/learning.py`；该脚本按自身位置定位项目，可用绝对路径从其他目录启动。
pip安装后也提供 `xyc-learning` 命令；从项目目录执行，或显式传入 `--config`。
源码脚本始终使用当前项目源码；普通pip安装的命令使用安装时版本，更新后重新安装即可。

把完整数据集目录复制到云端，包括 `meta/`、`data/` 和 `videos/`。
修改 `config/train/act.yaml` 的dataset/output_dir，或通过参数覆盖：

```bash
python tools/learning.py train --config config/train/act.yaml --dataset /data/my_dataset --output /data/runs/act_01
```

YAML中的相对路径按项目根目录解释；命令行覆盖的相对路径按当前终端目录解释。
首次正式训练会下载ResNet18的ImageNet视觉权重，约47MB，缓存于 `.setup/torch/`；也可设置TORCH_HOME。
这是图像骨干预训练权重，ACT任务策略仍需要用自己的数据训练。`--no-pretrained` 显式改为随机初始化视觉骨干。
权重恢复会加载checkpoint中的全部参数，不会再次下载视觉权重。

当前电脑的验证环境是CPU版torch，完整ACT已做1步测试，不代表GPU显存或速度测试。
需要本地重测流程时，用较小网络，约几秒计算加环境启动：

```bash
python tools/learning.py smoke --output outputs/act_smoke_02
python tools/learning.py smoke --config config/train/pi05.yaml --dataset /path/to/dataset --output outputs/pi05_smoke_01
```

`smoke` 默认在 CPU 上用缩小的网络、图像和动作块训练两步，然后回读权重做一次预测。π0.5 短训使用本地测试分词器和随机初始化的小模型，可离线验证流程。正式训练使用生产架构和对应的分词器。
输出目录已存在时拒绝覆盖；换新目录，或者对正式训练显式使用续训。

## 训练设置

默认完整ACT：ResNet18、512维Transformer、4层编码器、1层解码器、4层VAE编码器、latent32、KL权重10。
4相机分别以有序图像特征送入同一个模型；共享视觉骨干的可训练参数也更新，不冻结。
图像从视频解码为RGB，缩放到240×320，按ImageNet均值/标准差归一化。
状态和动作按训练回合的均值/标准差归一化，常量维度用epsilon避免除零；预测后只反归一化一次。
当前ACT不接收任务文字作为输入，任务文本仍保存在数据集中。

默认chunk_size=50，预测未来50个动作；n_action_steps=10表示推理队列每次使用前10步，再重新规划。
30Hz数据对应50步约1.67秒、10步约0.33秒。它不代表网络自身达到30Hz，也没有启动实时执行循环。
`runtime.PolicyRunner` 已实现每次取一个动作及回合reset；可选 `temporal_ensemble_coeff` 时要求n_action_steps=1。
原来的`xyc.learning.inference.PolicyRunner`导入仍保留为兼容别名，已有命令和记录代码无需修改。
实机控制频率还要结合GPU、网络和驱动测量。

默认训练10000步、batch8、AdamW、模型/视觉骨干学习率均1e-5、weight_decay1e-4、梯度裁剪10、FP32。
支持CUDA BF16混合精度，需要明确设置precision并由GPU支持；不静默退回其他精度。
batch、步数、学习率、调度器、warmup、验证/保存频率都在一个YAML里修改。
每1000步及最后一步保存；验证指标改善时另存最佳权重。验证用z=0推理的归一化动作L1，不称为任务成功率。
eval_batches控制验证预算，在整段验证回合中均匀取样，而非只检查开头。

回合不会随机拆成相邻训练帧和验证帧。当前5回合按seed42分为训练0/1/2/3、验证4。
可通过train_episodes和validation_episodes指定；validation_fraction=0表示全部训练，不再有独立验证结果。
当前回合2仅25帧，且此前数据含调试与中断，建议正式任务采集后重新筛选。
动作窗口到回合末尾会重复末动作并设置padding mask，loss忽略padding；不会读入下回合动作。
采样以训练step独立确定，Windows多进程预取不会改变断点采样游标。

## 保存与续训

```text
outputs/train/act_toys/
  run.json                         数据身份、模型/训练配置、标定、归一化和回合划分
  metrics.jsonl                    loss、KL、动作L1、数据/计算耗时、验证指标
  last_checkpoint.json             最近保存的checkpoint指针
  best_checkpoint.json             最佳验证checkpoint指针（有验证时）
  checkpoint-00001000/
    model.safetensors              全部模型参数与归一化缓冲量
    manifest.json                  格式、step、相机/维度/单位、统计量、资产和权重哈希
    training.pt                    AdamW、调度器、随机数和训练步数，用于续训
  result.json                      本次训练的最终结果
```

```bash
python tools/learning.py train --config config/train/act.yaml --resume outputs/train/act_toys
```

配置steps指训练总步数；常量学习率允许把10000改成20000再从10000继续，余弦调度续训需保留原总步数。
续训校验数据文件哈希、回合划分、相机/单位、模型和优化设置。数据可换目录，内容不能悄悄换掉。
相同CPU环境中，测试确认连续训练与2步保存后恢复到4步的所有权重逐元素一致；不承诺不同GPU后端跨设备逐位一致。
training.pt是PyTorch训练状态，只读取自己信任的项目checkpoint。
训练与数据集写入共用操作系统文件锁；异常退出后锁会自动释放，留下的锁文件可直接复用。
不要删除仍被使用的锁文件，否则可能产生两个不同 inode 的并发写入。

## 离线推理与实机边界

```bash
python tools/learning.py infer --checkpoint outputs/train/act_toys --dataset data/datasets/xyc/pick_toys_20261113_03 --episode 4 --frame 100 --output outputs/prediction.json
```

checkpoint可用具体checkpoint目录或整个run目录。run目录按已完整提交的checkpoint步数选择最新权重；
`last_checkpoint.json` 和 `best_checkpoint.json` 是从提交目录派生的索引；续训开始时重建，索引写入中断不改变已提交权重。
离线推理和部署可用 `--checkpoint-selection best` 从 run 目录按适配器声明的验证指标直接选最佳权重，不依赖索引文件内容；默认 `latest` 选择最新提交。
尚未提交的 `.checkpoint-*` 临时目录不参与选择；最新已提交目录损坏时明确报错，不悄悄回退。加载时检查权重SHA256。
验证时必须保持相机名称与顺序、状态/动作名称与单位、FPS及标定资产一致。
输出JSON包含50×18预测值、单位及首个动作的命名映射，没有电机写入。

| 下标 | state | ACT action输出 |
| --- | --- | --- |
| 0..13 | 左右臂标定位置 | 左右臂目标位置，非角度 |
| 14 | 升降高度m | 升降三值命令-1/0/+1 |
| 15..17 | 底盘实测vx/vy/wz | 底盘vx/vy/wz三值命令-1/0/+1 |

训练标签沿用现有“观测后新生成并被驱动接受的请求动作”，没有擅自改成下一帧state。
关节越界会拒绝；后四维只允许-1/0/+1，树莓派统一映射为升降0.015m/s、平移0.15m/s、旋转0.60rad/s。
ROS执行层校验标定、控制来源、时间戳/有效期和速度限制，并独占已有驱动接口；遥操与策略不能争用。
数据集、回放和模型运行时直接引用 `xyc.contracts` 中的18维动作定义。在线执行由统一控制通道转为 `SetArmTargets` 服务及底盘/升降速度话题，不另外维护未使用的控制消息。

## 推理运行记录

`infer` 默认自动记录，每次在电脑项目的 `outputs/inference/<UTC时间-随机编号>/` 新建独立目录。
原来的 `--output` JSON继续保留，记录第一个完整预测块；多帧运行的全部预测和选择过程在下面的Parquet文件中。
`--record-dir` 可以指定新的运行目录，已有目录拒绝覆盖。预测JSON不能覆盖meta、Parquet、事件或视频文件。

| 文件 | 内容 |
| --- | --- |
| meta.json | 模型、checkpoint路径及哈希、归一化、标定资产、任务、输入数据集路径/指纹、相机视频引用、运行结果和记录完整性 |
| predictions.parquet | 每次网络调用的prediction_id、观测引用、原单位state、完整H×D动作块和推理耗时 |
| steps.parquet | 逐步选择、真实下发结果、真实反馈三类记录，通过step_id关联；选择行另带prediction_id/chunk_index |
| events.jsonl | 开始、reset、异常、人工接管等调用者事件、结束；每条包含UTC和记录主机单调时间 |
| videos/相机名.mp4 | 开启视频后，每个被记录的选择步骤对应一帧，原始RGB观测/H264，包含配置中的所有相机 |

steps的record_kind分别为selection、execution、feedback：

- selection保存当前输入state、选中动作及其预测来源。它的sent_action和driver_accepted保持null。
- execution由实际执行层调用，记录真正提交的动作、驱动接受结果、命令编号和来源时间。accepted=null表示未确认，不当作成功。
- feedback由反馈层调用，保存之后测到的state；无效数值保存为null。

这三类数据是异步追加记录，不能把selection当作执行成功，也不能用旧反馈冒充执行结果。
普通动作队列会记录每一步来自哪次预测的第几个动作；时间集成会列出参与合成的所有prediction_id/chunk_index。
reset会清空待执行队列和集成历史，但运行内的日志编号不重新从1开始，避免关联到旧记录。
离线推理只生成prediction和selection；实时`deploy`通过同一记录器保存execution和feedback，并标明各自时钟来源。

示例：回放已有数据的30个观测，并保存四路画面，没有机器人控制：

```bash
python tools/learning.py infer --checkpoint outputs/act-full-size-check --dataset data/datasets/xyc/pick_toys_20261113_03 --episode 4 --frame 100 --frames 30 --record-video --output outputs/replay_prediction.json
```

Windows和Linux参数一致，使用安装ACT依赖的Python环境。
`--frames`默认1，不允许跨越回合结尾；回放使用原先录到的state，不是模型控制后的新实机反馈。
`--record-dir outputs/inference/my_test`指定新目录；`--no-record`关闭运行记录，但仍按`--output`保存预测JSON。
`--no-record`不能和`--record-dir`或`--record-video`同时使用。

离线默认不复制已有MP4，meta保存数据集身份、视频路径、回合和帧号，可以回溯原图。
`--record-video`另存的是模型输入的RGB尺寸（当前ACT是240×320），不是原相机640×480图像。
实时`deploy --record-video`保存对齐后的原始RGB观测，每路尺寸在一个运行中必须固定；不会复用归一化后的张量作为画面。
视频时间网格为video_frame_index/fps，steps保存帧索引和对应的观测引用。
视频容器30FPS不代表模型计算达到30Hz；推理耗时单独记录，离线来源时间明确标为数据集名义时间。
实际ROS时间应由调用者注明source_clock；不同主机的单调时钟不能直接相减。

公共接口如下。只做预测和记录时直接使用`PolicyRunner`；接实机时由`PolicyExecutor`
调用一个实现`CommandSink`的传输层，模型不能直接打开串口：

```python
from xyc.learning.inference_recording import InferenceRecorder
from xyc.runtime import PolicyExecutor, PolicyRunner

with InferenceRecorder.create(run_root, schema=adapter.schema,
                              metadata=run_metadata, record_video=True) as trace:
    runner = PolicyRunner(adapter, recorder=trace)
    action = runner.step(batch, observation=observation_reference)
    step_id = runner.last_step_id
    # 实际执行层拿到结果后，分别传入；本模块不负责发送命令。
    runner.record_execution(step_id, sent_action=actual_submitted_action,
                            accepted=driver_accepted, command_id=command_id,
                            source_time_ns=sent_ns, source_clock='robot_ros_time')
    runner.record_feedback(step_id, state=measured_state,
                           source_time_ns=feedback_ns, source_clock='robot_ros_time')
    trace.event('human_intervention', reason='operator took over')
```

`PolicyExecutor.step()`固定执行“模型预测→策略选一帧→校验18维平台单位→命名映射→CommandSink.dispatch”。
CommandSink返回`CommandResult`，再把真实接受结果和反馈写入同一条推理记录。
实时`RobotControlSink`通过17886控制通道提交动作；公共观测客户端从17883读取四路新JPEG，
并从17886状态接口取得带树莓派ROS时间戳的双臂、底盘和升降反馈。
ROS观测输出、录制和实时推理共用`src/xyc/observation.py`的`ObservationBuffer`：四相机时间差不超过33.3ms，
按图像组的中间时间选取邻近状态，检查关节采样时间、标定和控制会话，并拒绝重复图像。
这是读取时间配对，不是相机硬件同时曝光。ACT适配器只负责把配好的观测转换成ACT输入。
后续模型只需实现自己的`live_batch()`预处理和模型适配器，不复制硬件控制与保护。

树莓派ROS环境也需安装项目的轻量公共包：在项目根目录执行`python -m pip install -e . --no-deps`，
随后构建ROS工作空间。该包的默认依赖不包含PyTorch，纯观测配对不会导入GPU模型。

## 实时模型控制

先停止遥操或回放并支撑双臂，在树莓派启动唯一策略后端：

```bash
ros2 launch xyc_hardware policy.launch.py
```

电脑到树莓派的SSH连接需转发相机和控制端口：

```bash
ssh -L 17883:127.0.0.1:17883 -L 17886:127.0.0.1:17886 pi5@树莓派地址
```

在装有ACT和checkpoint的GPU电脑运行，默认先进入通用手动准备界面：

```bash
python tools/learning.py deploy --checkpoint outputs/train/act_toys --device cuda --record-video
```

准备阶段有两种输入方式。`--prepare-mode teleop` 使用本机两只已标定的主臂，对齐后自动接通双臂跟随。
`--prepare-mode keyboard` 只用键盘调底盘与升降，双臂保持当前位置。首次准备时两种模式都可以用键盘移动底盘与升降。
先按H完成缺失的升降回零，再用I/K调整高度、W/S前后、A/D平移、J/L旋转，空格停车。
遥操准备复用采集入口的主臂读取、坏帧处理、标定换算和绝对跟随映射。
按 P 开始推理。推理中按 P 后，机器人保持双臂当前位置并停车，模型动作队列清空。获取 TCP 发送锁后再次读键，丢弃暂停时尚未发送的模型动作。
遥操模式会显示主臂待调整的关节差值。连续三帧满足每个关节误差不超过 2 个标定单位、夹爪不超过 1 个单位时，
自动开始绝对跟随。从模型切到等待对齐时，底盘和升降保持停车。按 P 暂停后，等待对齐或已经接管时都可以按 R 恢复推理。Y/N 标注后按 P 开始下一轮。
同步模式每轮先做一次不下发动作的预热。RTC 在保持姿态时后台计算首块，新一轮只接收本轮启动的预测。
推理中Y标记本轮成功、N标记失败并返回准备；未标记不计为成功。Q退出评估。
切换时不重启驱动、不重新回零、不会自动升到某个任务高度。双臂保持上力，准备和模型使用同一个PolicyExecutor和机器人控制通道。
这是所有模型共用的准备流程，不是ACT内部逻辑。键盘字母大小写均可，Linux和Windows控制台共用入口。
主臂只在遥操准备阶段采样和生成手动目标；模型接管后暂停主臂采样，回到准备时恢复，始终只有一个命令来源。
无人值守调用需显式加`--auto-start`；此时仍可用`--home-lift`允许缺少高度参考时下行回零。

有限回归输出可能略超过标定端点（例如100.39）；公共执行边界将机械臂限制到完整[-100,100]、夹爪[0,100]范围，
不限制单步幅度、不使用额外端点余量。原始模型动作与实际下发动作分别记录，NaN/Inf仍拒绝。
人工示教数据和真实回放仍严格检查原始动作，不借此改写已录数据；电机自身及驱动的保护继续生效。

实时部署默认记录到电脑项目的`outputs/inference/<运行编号>/`。
`--record-dir`指定新目录，`--record-video`加存四路对齐画面，`--no-record`关闭记录。
正常结束、Ctrl+C和异常停止都会结束记录并保存运行结果；记录器缓冲有界，跟不上时标明缺失，不阻塞电机控制。
需要语言输入的适配器必须提供`--task "任务描述"`，程序会在连接并上力之前检查该要求。
人工评估结果与准备/接管时间写入events.jsonl，便于逐轮统计；准备阶段的键盘目标不冒充模型预测。

程序按checkpoint的FPS读取四路新图、18维反馈，调用ACT自己的实时预处理，再把动作送入与回放相同的
双臂、底盘和升降控制器。持续无法配出33.3ms以内的相机组、状态无法对齐、反馈/标定改变、推理严重落后、命令拒绝或连接中断都会停止。
该接口已接通代码链路，但新模型第一次实机运行仍需现场看护；checkpoint训练完成不等于任务已经安全验证。

记录器当前按单机器人batch=1保存。模型、归一化和标定版本等运行信息放入run_metadata。
程序正常结束的outcome=completed只表示运行结束，task_success默认null；任务成功应由调用者显式标注。
可用 `trace.close(task_success=True)` 结束并标注；不能根据程序没有报错推断任务成功。

磁盘写入和H264编码在独立线程中，生产者使用有上限的队列（默认256事件、64MiB排队数据）。
队列满时不等待磁盘，而是把缺失量记入meta；最终status=degraded，不能当作完整回放或训练数据。
磁盘/编码错误会停止继续记录，报告logging_failed及writer_error，保留已有文件，模型调用不会因该后台异常中断。
正常结束会等待已入队记录写完，关闭Parquet/MP4后去掉.partial后缀，并更新meta。
异常中断仍会尝试保存有效记录并标为failed；强杀/断电可能留下running状态和.partial文件，不能视作已完成。
记录会增加数据拷贝、CPU和磁盘负载；需要在真实部署机器上测量，不承诺打开视频后控制频率完全不变。

推理记录独立于人工示教数据，当前不自动导出或追加到LeRobot v3数据集。
后续需要先检查任务结果、人工接管与记录完整性，再把合格回合转成训练样本。

## 实机动作回放

这里的replay指让机器人重新执行已经采集的action，不加载模型、不读主臂，也不是播放视频。
当前支持本项目的18维整机数据：左右臂14个位置目标，以及升降、底盘vx/vy/wz的4个三值命令。
标定名称、单位、follower标定ID和动作时间约定必须匹配，读取的是action，不把下一帧state当动作。

| 文件 | 作用 |
| --- | --- |
| learning/replay.py | 电脑读取、预检查回合，按时发送动作；对齐初态、处理中断并保存执行日志 |
| learning/dataset_metadata.py | 训练与回放共用v3元数据/数值读取，回放不依赖PyTorch |
| xyc/transport/motion.py | 回放和实时推理共享的 TCP 客户端、协议和会话校验 |
| xyc_hardware/motion_replay.py | 树莓派ROS节点，将动作交给已有双臂服务、底盘/升降速度话题，并核对接受回执 |
| xyc_hardware/launch/replay.launch.py | 启动电机、双臂协调器及回放节点，command_source固定replay，不启动遥操控制器 |

原来的电机PID、过流/温度保护和标定换算不变。协调器允许replay使用已有批量写入通路，
不把数据伪装成主臂输入，也不绕过现有ROS驱动直接写串口。

先在电脑检查整段数据，此命令不连接机器人、不打开控制端口：

```bash
python tools/learning.py replay --dataset data/datasets/xyc/pick_toys_20261113_03 --episode 4
```

电脑仅需已有采集端依赖，不需要torch；也可以在项目目录安装 `python -m pip install ".[replay]"`。
Windows与Linux均可使用，保持完整项目目录（包含ros_ws里的公共协议），不要只复制单个脚本。

树莓派更新项目后构建一次：

```bash
conda activate xyc
cd ~/pi05_from_zero/ros_ws
colcon build --symlink-install --packages-select xyc_hardware
source install/local_setup.bash
```

给SSH增加回放端口转发，保持SSH连接。树莓派地址使用当前实际地址：

```bash
ssh -L 17886:127.0.0.1:17886 pi5@树莓派地址
```

切换前先支撑双臂：停止旧电机节点可能脱力。停止原遥操后端后，在树莓派启动：

```bash
ros2 launch xyc_hardware replay.launch.py
```

启动后只读取反馈，电脑的execute请求到来才上力。后端持有与遥操共用的运行锁，不能同时启动两套控制器。
现场确认运动范围和底盘路径清空，机器人底盘已摆到录制时相近的位置/朝向、能随时断电，再在电脑执行：

```bash
python tools/learning.py replay --dataset data/datasets/xyc/pick_toys_20261113_03 --episode 4 --execute --home-lift
```

Windows可使用 `python "$env:USERPROFILE\Desktop\实习准备\pi05_from_zero\tools\learning.py"` 这个绝对入口，
但`--dataset`的相对路径仍按当前终端目录解释；先进入项目目录，或给数据集绝对路径。
完整PowerShell命令也可以不使用自定义路径变量，避免变量尚未赋值时路径变成C:\tools：

```powershell
python "$env:USERPROFILE\Desktop\实习准备\pi05_from_zero\tools\learning.py" replay --dataset "$env:USERPROFILE\Desktop\实习准备\pi05_from_zero\data\datasets\xyc\pick_toys_20261113_03" --episode 4 --speed 1 --execute --home-lift
```

--home-lift明确允许在缺少高度零点时向下回零；已有有效零点时不重复回零，不加该参数则在缺零点时拒绝启动。
回零下行速度为10mm/s（当前机构约313编码器刻度/秒），不沿用原先1300刻度/秒的回零速度。
触底判定要求至少100ms的编码器位置窗口跨度不超过2刻度，且该窗口内持续存在至少0.3A的接触电流；
不再用单帧速度寄存器是否接近零来否定实际静止。接触后先停止并脱力，再检查至少1秒的位置稳定性，才建立零点。
单次电流尖峰、持续缓慢移动或仅仅无电流静止均不会建立零点。过流/过热、断联和行程保护保留。
这是现有接触式回零的改进，仍无法区分机械底部与中途障碍，操作前必须检查下行路径。
全行程回零超时按行程和速度计算，默认约70秒；电脑最多等待90秒。
底盘/升降使能前要求至少150ms的连续完整编码器反馈，各轴位置窗口跨度不超过2刻度；
单帧速度±50或moving标志抖动不再单独导致静止检查失败，真正持续位移仍拒绝使能。
回放后端会短暂等待这一窗口就绪。寄存器读取缺回复或扭矩字节无效时最多做3次只读请求，
舵机明确告警立即报告，运动和使能写入不盲目重发。
部分使能失败会清理本节点已经尝试上力的轴；脱力未确认时保留对应ID和故障，
不会把未知来源的上力状态自动认领为本节点控制。

上力和回零前会根据当前反馈预检查录制起点。连续绝对位置控制统一使用完整标定范围：关节[-100,100]、夹爪[0,100]。
遥操和回放均不保留额外2单位的端点余量；例如左肘97.800到99.358是合法目标。
连续控制不再按24/32单位关节超前量、10单位夹爪超前量截短目标，也不按旧的逐帧步幅拒绝合法绝对目标。
遥操上层的目标变化率限制、超前量限制和提前减速也已移除；绝对跟随保留最新主臂目标，
相对跟随只执行既有起点偏移映射。无效或越界目标明确拒绝，不默默裁剪。
持续控制只有一套舵机速度配置：普通关节`2400/120`、夹爪`2000/100`（速度/加速度寄存器值）；
电机反馈目标60Hz，主臂请求70Hz。摇操与回放不再有可选慢档；实际机械速度仍受负载和舵机能力影响。
编码器跨圈、过流/过热、通信与标定身份检查继续保留。

遥操与回放共用同一套`JointControl`和连续位置安全规则；
连续夹爪在共享的`JointControl`中单独处理，遥操与回放不各自实现一套。
`0.3 A`和`load_raw=300`作为接触候选，不再因单帧超限就脱力；负载持续且编码器约120ms无进展时，
夹爪将目标收回到当前位置附近8个编码器刻度，保持上力，并锁住继续闭合的目标；明确反向张开后解除锁存。
夹爪接触判定和机械堵转判定要求相邻反馈不超过60ms。电气过流使用公共 `CurrentProtection`，
双臂、夹爪、底盘和升降共用计时实现：在允许的200ms反馈间隔内持续累计，10Hz反馈也会正常触发。
低于相应电流阈值的新反馈会清除该计时；高电流前后间隔达到200ms时按保护监测丢失处理，不反复清零后继续上力。
上力前会把STS寄存器48设为可配置扭矩上限，默认`100/1000`（10%最大输出扭矩），值由`gripper_torque_limit_raw`统一配置；
这是舵机输出扭矩比例，不是指尖牛顿力。持续电流达到3.8 A约200ms、温度/反馈/通信异常仍会保护脱力。
单个无效关节反馈帧不会立即脱力：底层先保持旧目标，连续丢失达到200ms才保护脱力；遥操层在此期间暂停发送新目标，
反馈恢复即可继续，持续无效500ms才停止遥操控制器。真实越界数值立即保护；持续舵机告警会作为反馈持续无效触发200ms保护。
这个10%是首次调试起点，需按实际夹爪和物体逐步校准，不能当作已测得的安全夹持力。

执行顺序是：校验整段动作 → 检查当前反馈和标定 → 双臂原位上力、底盘使能 → 必要时回零 →
按单独限定的起始对齐速度移动到录制首帧的14维双臂位置和升降高度 → 按记录时间执行选定回合 → 底盘/升降发零速度，双臂保持最后目标。
对齐本身会移动机器人，不能只给录制轨迹留空间；目前双臂对齐约8标定单位/秒，升降最多7.5mm/秒。
开始姿态无法在120秒内达到会停止；它不是碰撞规划器，不会自动绕开障碍。

默认使用已有observation.source_time间隔，例：900帧约30.55秒；缺少该列的数据才使用fps网格。
时间列异常时拒绝执行，不默默替换。底盘/升降只有-1/0/+1固定速度，
因此包含这四维非零动作的回放只允许`--speed 1`；不会生成0.5命令来假慢放。
只有底盘/升降全程为0的手臂片段才能使用小于1的`--speed`拉长时间。
`--frame`、`--frames`可限定同一回合内的片段；选中的首帧会成为新的对齐起点。
手臂专用片段的时间倍率仅允许0.1～1.0。保持现有驱动上限，不把数据中的真正越界目标裁成可执行目标；
仅将float32在速度上限附近约1e-8～1e-7的表示误差恢复为原上限值。

每次target用新的序号和树莓派当前时间，不重放旧网络序号/过期时间戳。
两条手臂总线批量提交，随后提交4轴速度并等待驱动接受事件；不宣称18轴硬件原子下发。
任一目标被拒绝、回执不明或持续跟不上时停止，不跳过数据帧追赶，也不盲目重发结果不明的命令。
数据间隔中由控制循环发心跳维持指令。电脑另有每 80 ms 的只读状态查询，用于保持 TCP 连接。只读查询不延长运动期限。运动过程中超过 0.5 秒没有新目标或控制心跳，后端触发保护停车。原位保持确认后有 5 秒等待窗口，用于首次准备和同步模型预热。RTC 等待首块时由运行循环发送心跳。
后端在等待 ROS 回执期间也会检查 TCP 断连。发现断连后，取消尚未发出的目标并向两条总线请求保护停车。电机节点先锁存故障，再清理串口上的运动状态。通信失败时保留诊断信息。

执行日志在电脑 `outputs/replay/<运行编号>/`，可用`--output`指定新的目录：

- meta.json：数据身份、标定、时间方式、速度倍率、完成/失败原因、已接受帧数和停止确认情况。
- steps.jsonl：对齐/回零/正式回放阶段，原数据帧号、实际发送动作、驱动接受结果、往返耗时及实测反馈快照。
失败时另存failure记录及meta.json中的failure_snapshot；后端拒绝回执附带当时反馈和故障原因。
标定变化与反馈/底层保护停止分别报告，升降持续过流会显示具体ID、电流和阈值，不再统称为标定变化。
后端拒绝一次已启动的请求时，先执行停止清理，再随拒绝回执返回stop_confirmed及stop_error。
电脑显示并记录停车确认状态及具体失败原因。连接已断时，后端保留诊断，之后可通过只读状态查看。

反馈快照包含其来源接收时间；目标被接受不等于每个采样点已物理到位。
反馈快照新增driver_targets、driver_target_ticks和driver_target_stamps_ns：
它们来自最新底层状态样本，记录最后已写入并回读确认的目标；可能比当前RPC回执晚一个采样周期，
应结合样本时间比较。它们与action请求以及state实测位置分别保存，不再只根据accepted判断目标值是否一致。
日志写入失败或活动中的网络连接断开时，遥操、回放和推理共用电机保护停车入口。底盘和升降写零速，双臂释放扭矩，故障会话锁定。正常 Ctrl+C 会尝试停车并保持双臂当前位置。停车未确认时，回放日志保留 `stop_confirmed=false`。
不要在回放中手动移动机械臂，正式运行前需在现场完成验证。

自动验证包括PC协议/流程测试及树莓派隔离ROS测试（使用假总线和假底盘）。
旧版比例速度契约下的实机日志不能作为当前三值命令契约的验证结果；
新版需同步到树莓派并重新构建后再做实机验收。

执行期限由公共 `PolicyExecutor` 检查。一次实时观测、预处理和预测若超过 0.25 秒，
预测结果不会下发。剩余有效期随 v5 控制协议传到树莓派。底盘和升降经单个 `MobileCommand` 携带相同绝对期限，接收端、周期执行和串口写入前均检查。升级后需同步并重新编译 `xyc_interfaces` 与后端，
协议版本不匹配时会在准备阶段拒绝控制，不采用旧目标重试。

动作最后四维始终是 `-1/0/+1`，状态保留实际物理单位。各轴名义速度从公共契约读取；
平移与旋转的组合超过轮速上限时，底层统一按相同比例缩小三个机体速度，保留方向和旋转比例。
录制动作仍保存原三值指令，实测速度保存在 state。数据读取器不再自动把旧速度或比例动作改写成三值。

回合提交先持久化日志和元数据备份，再移动数据、视频和统计文件，最后更新 `info.json`。
若提交中断，恢复同一树莓派缓存时：已提交回合保留并返回保存回执；未完成回合先回滚其文件和统计，
然后重传。正在运行的客户端可按 B 重试；重启客户端使用同名配置和 `--resume`。
