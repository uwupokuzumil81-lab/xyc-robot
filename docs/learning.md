# ACT：训练、保存与推理

当前已完成模型侧的闭环：现有 LeRobot v3 数据 → 四路图像与18维状态 → ACT训练 → 平台checkpoint → 重载 → 50×18动作预测。
正式长训练尚未启动；测试权重仅验证计算流程。`learning.py deploy` 已通过公共运行时和回放控制后端接到同一套18维实机控制，
但还没有用有效训练权重完成实机策略验证；它当前是电脑/GPU进程经本机转发控制ROS后端，不是树莓派内的独立ROS策略节点。
云服务器推理传输也尚未实现，不能把离线测试权重当成已经能完成任务的模型。

## 哪些统一，哪些由模型决定

| 环节 | 平台公共代码 | ACT 专用代码 |
| --- | --- | --- |
| 读取数据 | learning/dataset.py：v3索引、视频解码、回合划分、动作窗口 | 通过模型配置指定图像尺寸、窗口长度 |
| 配置 | learning/config.py：路径、训练步数、优化器、保存频率 | models/act/configuration_act.py：ACT网络、VAE、chunk参数 |
| 输入输出 | learning/contracts.py：相机名、状态/动作顺序、单位 | models/act/adapter.py：图像归一化、张量打包、反归一化 |
| 训练 | learning/trainer.py：优化、验证、日志、续训 | ACT提供loss和参数组，网络代码在modeling_act.py |
| 保存恢复 | learning/checkpoint.py：权重、资产、配置、统计量、训练状态 | 无另一套LeRobot保存/恢复接口 |
| 推理 | runtime/：逐帧执行、策略选择、18维命令映射；learning/inference.py负责离线数据入口 | ACT输出动作chunk，可选时间集成 |
| 机器人驱动 | 原有ROS功能包 | 模型不能自行打开串口或绕过驱动 |

以后接模型，新增 `models/<name>/` 的配置、模型、适配器，在 `learning/registry.py` 注册并给一个训练YAML。
适配器提供 `compute_loss`、`predict_chunk`、`validation_metrics`、`optimizer_groups`、`reset`，以及窗口/图像配置和统计量。
模型需要的语言、历史观测或特殊编码应在其适配器中明确处理；公共契约需扩展时统一扩展，不能假设所有模型都和ACT完全一样。

运行时已拆成三层：

```text
PolicyRunner                 调模型并记录预测，交给选定的推理策略
  └─ strategies/
       ├─ sync_chunk.py      普通动作块队列；ACT可选时间集成
       └─ rtc.py             RTC统一接口，只接受真正提供RTC后端的模型
PolicyExecutor               把选中的18维动作映射成统一机器人命令并交给CommandSink
```

RTC不是对所有动作块都能套用的通用平滑函数。公共层只统一调度、动作选择和执行接口；
Π0.5的引导去噪、上一动作块条件、推理延迟和execution horizon属于模型专用实现，
以后放在`models/pi05/`并通过`create_rtc_runtime()`接入。当前ACT没有该能力，显式选择`rtc`
会被拒绝，不会把普通异步队列冒充RTC。`PolicyExecutor`也不包含串口、ROS或安全阈值；
这些仍由一个平台命令接收端转交现有统一控制器。

ACT源码固定自 LeRobot v0.4.3；`models/act/PROVENANCE.json` 保存上游地址、原文件哈希和修改说明。
修改仅是本地配置、常量和基类依赖接线；网络、VAE和损失计算保留该版本实现。Apache-2.0许可证随模型保存。
运行时模型不导入 `lerobot`，公共训练器也不使用其配置类或处理器工厂。

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
```

`smoke` 会缩小网络、图像和chunk，训练2步，再从权重重新加载预测。不能把这份配置当作完整ACT训练结果。
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
意外强杀留下.training.lock时，确认该目录没有训练进程后再移除锁；不能同时往同一训练目录写入。

## 离线推理与实机边界

```bash
python tools/learning.py infer --checkpoint outputs/train/act_toys --dataset data/datasets/xyc/pick_toys_20261113_03 --episode 4 --frame 100 --output outputs/prediction.json
```

checkpoint可用具体checkpoint目录或整个run目录（读取last指针）。加载时检查权重SHA256。
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
`RobotCommand` 是这套18维动作定义的 ROS 消息镜像；当前在线执行实际走本机回环控制通道。数据集、回放和模型运行时均使用同一动作定义：前14维目标位置，后4维三值命令。

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
| videos/相机名.mp4 | 开启视频后，每个被记录的选择步骤对应一帧，RGB/H264，包含配置中的所有相机 |

steps的record_kind分别为selection、execution、feedback：

- selection保存当前输入state、选中动作及其预测来源。它的sent_action和driver_accepted保持null。
- execution由实际执行层调用，记录真正提交的动作、驱动接受结果、命令编号和来源时间。accepted=null表示未确认，不当作成功。
- feedback由反馈层调用，保存之后测到的state；无效数值保存为null。

这三类数据是异步追加记录，不能把selection当作执行成功，也不能用旧反馈冒充执行结果。
普通动作队列会记录每一步来自哪次预测的第几个动作；时间集成会列出参与合成的所有prediction_id/chunk_index。
reset会清空待执行队列和集成历史，但运行内的日志编号不重新从1开始，避免关联到旧记录。
当前离线回放只生成prediction和selection；execution、feedback接口等待ROS实际执行层接入，不会自动编造它们。

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
未来实时执行层可以通过video_frames参数提交原始RGB图像，每路尺寸在一个运行中必须固定。
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
实时`RobotControlSink`通过17886控制通道提交动作；ACT适配器从17883读取四路新JPEG并完成专用预处理。
后续模型只需实现自己的`live_batch()`预处理和模型适配器，不复制硬件控制与保护。

## 实时模型控制

先停止遥操或回放并支撑双臂，在树莓派启动唯一策略后端：

```bash
ros2 launch xyc_hardware policy.launch.py
```

电脑到树莓派的SSH连接需转发相机和控制端口：

```bash
ssh -L 17883:127.0.0.1:17883 -L 17886:127.0.0.1:17886 pi5@树莓派地址
```

在装有ACT和checkpoint的GPU电脑运行；缺少升降零点时才显式加`--home-lift`：

```bash
python tools/learning.py deploy --checkpoint outputs/train/act_toys --device cuda --home-lift
```

程序按checkpoint的FPS读取四路新图、18维反馈，调用ACT自己的实时预处理，再把动作送入与回放相同的
双臂、底盘和升降控制器。相机时间差超过50ms、反馈/标定改变、推理严重落后、命令拒绝或连接中断都会停止。
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
| xyc_hardware/replay_transport.py | 两端共用的协议、动作维度/限速校验、独占会话和顺序检查 |
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
接触和堵转的持续时间只累计相邻有效反馈；相邻反馈间断超过60ms会重新计时，避免把通信空档误算成持续受力。
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
数据间隔中有心跳续租速度指令；连接断开后，后端尝试发零速度，并让双臂保持当前测得位置。
底层速度驱动也有命令超时保护；网络/程序停止不能代替现场断电。
电脑另外每80ms在独立线程中只读查询status，保持TCP连接；它不发送target，也不续租速度指令。
所有请求共用互斥的发送/接收通道，保活不会抢走动作回执。这样回零结束后的准备、打印等短暂停顿不会误触发350ms连接空闲超时，
而主控制循环停顿时，底盘/升降仍保留原来的运动命令超时停车机制。通信中断转为明确错误，不自动重连、重发动作。

执行日志在电脑 `outputs/replay/<运行编号>/`，可用`--output`指定新的目录：

- meta.json：数据身份、标定、时间方式、速度倍率、完成/失败原因、已接受帧数和停止确认情况。
- steps.jsonl：对齐/回零/正式回放阶段，原数据帧号、实际发送动作、驱动接受结果、往返耗时及实测反馈快照。
失败时另存failure记录及meta.json中的failure_snapshot；后端拒绝回执附带当时反馈和故障原因。
标定变化与反馈/底层保护停止分别报告，升降持续过流会显示具体ID、电流和阈值，不再统称为标定变化。
后端拒绝一次已启动的请求时，先执行停止清理，再随拒绝回执返回stop_confirmed及stop_error。
电脑保留这个确认结果，避免清理已成功却因后续连接关闭被误报为停止未确认。

反馈快照包含其来源接收时间；目标被接受不等于每个采样点已物理到位。
反馈快照新增driver_targets、driver_target_ticks和driver_target_stamps_ns：
它们来自最新底层状态样本，记录最后已写入并回读确认的目标；可能比当前RPC回执晚一个采样周期，
应结合样本时间比较。它们与action请求以及state实测位置分别保存，不再只根据accepted判断目标值是否一致。
日志写入失败会停止动作回放。Ctrl+C中断电脑客户端会尝试停车并冻结双臂当前目标；停车未确认时日志保留stop_confirmed=false。
不要在回放中手动移动机械臂，正式运行前需在现场完成验证。

自动验证包括PC协议/流程测试及树莓派隔离ROS测试（使用假总线和假底盘）。
旧版比例速度契约下的实机日志不能作为当前三值命令契约的验证结果；
新版需同步到树莓派并重新构建后再做实机验收。
