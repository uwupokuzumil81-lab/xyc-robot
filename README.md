# XYC：从零搭建的机器人 ROS 2 项目

电脑与树莓派保存同一份项目源码、主臂标定和从臂标定，根据硬件连接运行不同部分。
主臂插在电脑，所以电脑采集主臂位置和键盘；从臂、相机、底盘及升降连接树莓派，所以树莓派执行驱动和 ROS 节点。
电脑支持 Windows / Linux，日常控制只使用下面这一套遥操入口。

## 项目结构

```text
pi05_from_zero/
├── README.md
├── requirements.txt             电脑端 Python 依赖
├── environment.yml              树莓派 xyc / ROS 2 Jazzy 环境
├── config/
│   ├── leader_calibration/      左右主臂标定
│   └── calibration/             左右从臂标定
├── tools/
│   ├── teleop_stream.py         电脑端统一遥操：双臂读取和底盘/升降键盘
│   └── leader_arm.py            主臂标定、只读检查；也供遥操导入
└── ros_ws/src/
    ├── xyc_interfaces/          项目消息及双臂控制服务定义
    ├── xyc_camera/              四路相机节点、配置及启动文件
    ├── xyc_hardware/            电机驱动、标定、遥操接收及执行
    ├── xyc_observation/         四幅图像与18维整机状态的统一观测
    └── xyc_recording/           树莓派回合缓存、结束后传输、电脑LeRobot v3保存
```

`ros_ws/build`、`install`、`log`、`.setup` 和采集的 `data` 是各台机器的本地产物，
不当作源码互相覆盖。尤其不能把 Windows 的二进制依赖复制到树莓派使用。
标定文件属于机器人资产，两端源码同步时也应同步；重新标定后，以实际连接硬件的一端新保存的文件为准。

## 安装与构建

电脑在项目目录执行，不需要安装 ROS：

```bash
python -m pip install -r requirements.txt
```

树莓派首次安装使用 `environment.yml` 创建 `xyc` 环境；已有环境无需重复创建。
在树莓派上构建整套 ROS 功能包：

```bash
conda activate xyc
cd ~/pi05_from_zero/ros_ws
colcon build --symlink-install
source install/local_setup.bash
```

每个新的树莓派终端都需要先激活环境，再加载工作空间。
若旧终端已有 ROS 的 `PYTHONPATH` 并出现 Conda/OpenSSL 混用报错，先 `unset PYTHONPATH`，
再激活环境、重新加载工作空间。

## 日常启动遥操

先确认机器人供电、机械臂支撑和运动空间。当前模式会在主臂连接后使从臂上力，
再逐步向主臂的绝对标定位姿跟随；两边开始时的姿态可能不同。

电脑开一个终端，建立 SSH 转发并保持这个窗口开启。下面是当前确认可用的有线连接示例；
实际使用时以当前能登录的树莓派地址为准，IPv6 地址后的 `%9` 也是电脑本机的网卡编号，换网卡后需要更新：

```powershell
ssh -6 -o ExitOnForwardFailure=yes -L 17882:127.0.0.1:17882 -L 17883:127.0.0.1:17883 "pi5@fe80::7c92:ee22:7671:c10f%9"
```

进入树莓派后启动统一后端。它会启动左右电机节点、双臂协调器、主臂接收桥和遥操控制器，
不需要再单独启动电机或键盘控制程序：

```bash
conda activate xyc
source ~/pi05_from_zero/ros_ws/install/local_setup.bash
ros2 launch xyc_hardware teleop.launch.py side:=both live:=true with_motors:=true follow_mode:=absolute chassis_mode:=true
```

电脑另开一个终端，在项目目录启动唯一的遥操客户端：

```bash
python tools/teleop_stream.py leader --side both --keyboard
```

Windows PowerShell 可以先执行 `cd "$env:USERPROFILE\Desktop\实习准备\pi05_from_zero"`。
Linux 电脑使用自己保存的同一项目目录。主臂串口按 USB 序列号识别，无需固定 COM 编号。
启动失败时先检查树莓派后端和 SSH 转发，重复启动电脑程序不能修复未监听的端口。

## 四路相机窗口

统一遥操后端默认同时启动四路相机和只读预览服务；`with_cameras:=false` 可关闭这部分。
如果相机已经由其他启动文件运行，应避免再次启动同一批相机。
电脑客户端启动遥操后默认自动在浏览器打开 `http://localhost:17883`，
以四宫格显示前方、胸前、左腕和右腕画面。也可以手动打开这个地址。
电脑无需安装 ROS 或额外的视频客户端，但 SSH 必须同时转发 17882 和 17883 两个端口。
Windows 与 Linux 桌面电脑都调用本机默认浏览器；需要关闭自动弹窗时可加 `--no-preview`。
电脑端使用 `localhost`，同时兼容 SSH 的 IPv4 和 IPv6 本地监听；预览就绪检查直接访问本机，不经过系统 HTTP 代理。
统一启动会先取得后台运行锁并检查旧组件；已有后台时直接拒绝第二次启动，
不会先启动部分相机或电机再报端口冲突。退出整套后台后才能重新启动；电脑客户端可单独退出再连接。
连续跟随中，已确认“目标过期且未写入”的单侧拒绝不会让双臂脱力；即使另一侧已接受，
也不把这组不完整请求作为14维训练动作保存，而是从下一组新反馈重新规划。
总线执行结果不明、电流/温度保护和持续反馈中断仍停止，不通过重复旧目标掩盖故障。

预览默认目标 30 FPS、JPEG 质量 75；原始图像话题仍按相机设置的 30 FPS 请求采集。
实际频率受相机和机器负载影响。JPEG 编码在相机节点的独立线程进行，只保留最新帧；
没有浏览器观看时，预览节点取消图像订阅，相机停止额外的 JPEG 编码。
这些设置由相机节点的 `preview_fps`、`preview_quality` 参数控制。
画面超时会遮罩提示，避免把旧画面误认为实时画面；双击单幅图像可以全屏。
浏览器只显示图像，不发送控制命令。键盘控制底盘和升降时，先把焦点切回遥操终端。

相机功能包新增 `xyc_camera/preview_node.py`（ROS 压缩图像转浏览器预览）、
`web/cameras.html`（四宫格界面）和 `test/test_preview.py`（预览与原始图像兼容检查）。

## 操作按键

字母大小写均可。手臂直接跟随主臂；以下键盘指令只控制底盘和升降。

| 按键 | 功能 |
| --- | --- |
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移 |
| J / L | 左转 / 右转 |
| I / K | 升降上行 / 下行，先完成回零 |
| H | 升降下行回零，先确认底部路径畅通 |
| 空格 | 停止底盘和升降，包括取消正在进行的回零 |
| Q | 停止底盘和升降并退出电脑遥操客户端 |

连接后自动以零速度准备底盘和升降，不需要按 E。升降每次重启驱动后需重新建立高度零点。
回零期间显示“回零中 / 等待回零稳定”，忽略方向要求但保持通信，完成后才能用 I/K。
回零期间重复按 H 只提示当前进度，不会重新启动回零。高度测量在上下边界保留
10 个编码器刻度的容差（当前丝杆导程下约 0.32 mm），显示值限制在合法行程内，
内部保留原始累计位移；超过容差仍保护停止，升降运动目标和软件停车边界不放宽。
Windows 可以识别按住和松开；Linux 终端依赖按键重复，约 0.25 秒无按键后停止该方向。
命令中断会触发底盘/升降停车。Q、空格及网络软件停止都不能代替现场断电措施。
电脑客户端退出后，双臂驱动仍可能保持最后目标；停止电机节点则会尝试脱力，需先支撑双臂。

## 标定、相机与数据

主臂标定必须在连接主臂的电脑上执行；这是维护工具，不是另一个运动控制入口：

```bash
python tools/leader_arm.py left calibrate
python tools/leader_arm.py right calibrate
```

从臂标定通过树莓派的 `ros2 run xyc_hardware calibrate_arm` 工具完成。
统一遥操已经启动相机时，不要再单独启动 `cameras.launch.py`。
只使用相机时可运行 `ros2 launch xyc_camera cameras.launch.py`；
观测使用 `ros2 launch xyc_observation observation.launch.py`，默认复用已经存在的输入。
观测schema v2是四幅图像与18维整机状态：双臂14维、升降高度、底盘实测vx/vy/wz。最后四维是带物理单位的状态，不是机械臂关节角，也不是-1/0/+1动作。

项目包含机器人驱动、遥操、ROS 观测和 LeRobot v3 数据采集。
已接入 ACT 的数据读取、训练、断点续训、权重保存和离线推理；Π0.5 仍待接入。
已有云服务器模型工程不属于这里的源码。

## ACT 与统一训练框架

ACT 使用收进项目的 LeRobot 0.4.3 模型实现，运行时不依赖另一份 LeRobot 仓库。
训练和模型推理单独运行在电脑或云 GPU，不在树莓派的 ROS 环境里安装 PyTorch。
ACT专用适配器负责把四路实时JPEG转换成训练一致的240×320 RGB张量；
公共运行时把18维输出交给统一控制后端，不直接写串口。`deploy`可连接真实机器人。

```text
src/xyc/learning/        公共数据、配置、训练、checkpoint、推理入口
src/xyc/models/act/      ACT 模型、专用配置、输入输出适配、原始许可证
src/xyc/runtime/         公共逐帧执行器、同步动作块策略、RTC扩展接口
config/train/act.yaml    ACT 训练设置
tools/learning.py        统一命令：inspect / smoke / train / infer / deploy
outputs/                各次训练的权重、日志、预测结果，不打进源码包
```

在独立 GPU 环境中，从项目目录安装 `python -m pip install ".[act]"`，然后运行：

```bash
python tools/learning.py inspect --dataset data/datasets/xyc/pick_toys_20261113_03
python tools/learning.py train --config config/train/act.yaml
python tools/learning.py infer --checkpoint outputs/train/act_toys --dataset data/datasets/xyc/pick_toys_20261113_03 --episode 4 --frame 0 --output outputs/act_prediction.json
```

训练配置默认四相机、18维、10000步、batch 8、学习率1e-5、每1000步保存。
按完整回合划分验证集，统计量只用训练集计算；默认使用240×320 RGB图像。
这些是可编辑的初始设置，不代表当前5回合调试数据足够学会任务。
完整操作、续训、动作语义及实机部署边界见 [ACT 使用说明](docs/learning.md)。

离线 `infer` 现在默认生成独立推理记录目录 `outputs/inference/<运行编号>/`，
保存模型/数据身份、完整动作块、逐步选中动作、耗时和事件；`--frames 30` 可连续回放30帧。
`--record-video` 额外保存四路模型输入画面，默认引用已有数据视频；`--no-record` 只保留原来的预测JSON。
执行和反馈有独立记录接口，离线测试的下发／接受状态保持未知，不会被标成已控制机器人。

`runtime.PolicyExecutor`规定所有模型最终输出同一个18维平台动作契约，再交给统一`CommandSink`；
它不直接写ROS、串口或电机。`sync_chunk`是ACT当前使用的普通动作块/时间集成策略。
`rtc`只是一层统一扩展接口，真正的Π0.5 RTC需由后续`models/pi05/`实现引导去噪和动作块衔接；
ACT没有该能力，不能把普通异步推理称为RTC。

后四维只允许`-1/0/+1`三种命令，不是比例和档位：`+1`正向固定速度，`0`停止，`-1`反向固定速度。
固定映射为升降0.015m/s、底盘前后/平移0.15m/s、旋转0.60rad/s。模型的连续输出在执行边界按±0.5阈值量化为三值命令，不会向机器人发送0.5。
关节前14维仍是绝对标定位置。实机启动方式见[ACT使用说明](docs/learning.md#实时模型控制)。

## 采集动作的实机回放

`learning.py replay`读取已采集回合的18维action，按记录时间通过ROS驱动执行，不需要ACT模型和主臂。
它与`infer --frames`不同：后者把旧观测送进模型，本功能直接把录制动作交给机器人。
新增ROS后端为 `ros2 launch xyc_hardware replay.launch.py`，与遥操后端互斥；启动后不自动上力。
电脑显式加`--execute`才会连接并上力，先缓慢对齐首帧双臂和升降，再开始这一回合。
底盘仍需人工放到录制时相近的位置和朝向；数据没有底盘全局位姿，按轮速回放会受打滑影响。
完整启动、SSH转发和停止方式见 [实机动作回放说明](docs/learning.md#实机动作回放)。

## 遥操同时采集 LeRobot v3

原来的遥操命令保持可用：不加 `recording:=true` 和 `--record` 就不录制。
数据集默认保存在运行遥操程序的本地电脑，路径为项目下 `data/datasets/<数据集名称>/`。
树莓派按原始采集时间对齐四路图像、状态和新动作，录制期间顺序写入本地回合缓存。
达到指定帧数或按B结束后，才通过独立数据连接传到电脑；电脑写盘、编码视频并确认成功后，这一回合才算完成。
数据通道复用现有17883 SSH转发，不与17882的遥操请求共用连接；电脑不需要安装ROS、PyTorch或LeRobot。
电脑安装依赖：`python -m pip install -r requirements.txt`。树莓派更新源码后重新构建：

```bash
conda activate xyc
cd ~/pi05_from_zero
cd ros_ws
colcon build --symlink-install --packages-select xyc_interfaces xyc_recording xyc_hardware --cmake-force-configure
source install/local_setup.bash
```

保持原来的两个 SSH 端口转发。树莓派后端加 `recording:=true`：

```bash
ros2 launch xyc_hardware teleop.launch.py side:=both live:=true with_motors:=true follow_mode:=absolute chassis_mode:=true recording:=true
```

Windows 电脑示例：每回合900个有效样本，名义30秒，最多录50回合：

```powershell
python "$env:USERPROFILE\Desktop\实习准备\pi05_from_zero\tools\teleop_stream.py" leader --side both --keyboard --record --dataset xyc/pick_toys --task "Pick up the toy and put it in the box" --fps 30 --episode-frames 900 --episodes 50
```

Linux 桌面在项目目录使用同样参数：

```bash
python tools/teleop_stream.py leader --side both --keyboard --record --dataset xyc/pick_toys --task "Pick up the toy and put it in the box" --fps 30 --episode-frames 900 --episodes 50
```

`--episode-seconds 30` 可代替 `--episode-frames 900`；时长按帧数除以目标FPS换算。
相机实测29.5Hz时，采足900个新样本约需30.5秒，不会用旧图补满30Hz。
`--data-dir "D:/robot_data"` 可更改本地电脑的数据根目录；Linux例如 `--data-dir /mnt/data/robot`。
该参数是电脑路径，不是树莓派路径。终端启动时会打印数据集的实际绝对保存位置。
旧树莓派上的已保存数据不会删除、覆盖或自动混入本地数据集；需要追加时先完整复制旧数据集，再使用`--resume`。
不指定 `--dataset` 时按电脑当前时间新建名称；同名数据集默认拒绝覆盖，追加必须显式加 `--resume`。
追加时要求FPS、维度、图像尺寸、标定资产和动作时间约定一致。`--episodes 0` 表示手动不限回合数。
旧版“观测时已生效目标”的数据不与新版混合追加，已有文件不改写；新版请使用新的数据集名称。

启动后先待机，不自动开录。完成主从臂、四路相机和底盘准备后按 **R** 开始第一回合。
整机18维模式若显示“未回零”，先按 **H** 完成升降回零；已经显示有效高度就不必重复回零。
未知高度不会作为0写入。录制回合打开后，H会提示先B保存或N丢弃，避免回零打断数据；I/K升降照常使用。
达到帧数后自动保存，等待人工复位。
复位期间遥操仍正常，数据不写入；复位完成按R开始下一回合。
采集遇到短暂缺帧会等待新数据，不重复旧图或旧动作。有效数据连续中断超过0.5秒时，
自动保存已经录到的有效前段，显示具体缺失项，并在回合质量文件的 `interruption` 中记录原因；
恢复遥操后按R开始新回合，不把长时间中断前后的动作直接拼成连续视频。
串口写入使用独立50ms等待预算，正常收发仍立即返回，不改变60Hz反馈目标。
目标提交前的读取请求发生偶发写入超时时补读一次；运动/扭矩写入不盲目重发，持续通信故障仍停止。
同步读取反馈不完整时也补读一整轮，不拼接前后两轮制造完整状态。底盘/升降若仍缺回复，
先发零速度并暂时禁止运动；0.2秒内完整反馈恢复后等待新的速度指令，保留仍可连续追踪的升降零点。
持续缺失达到0.2秒、零速度无法确认或电机明确报警时保护停止，录制不使用缺失期间的旧状态。
遥操启动上力仍走双总线准备/提交；进入连续跟随后，每条目标在各总线上执行一次
新鲜反馈校验和批量写入，省去每帧重复的准备读取及第二轮服务往返。电流、温度、
标定、控制所有权、目标范围和命令有效期仍检查，两条独立总线不宣称硬件原子提交。
保存提示中的“实际FPS”和质量文件 `fresh_fps` 才是有效采样率，视频容器的30FPS是配置值。
连续控制下，遥操控制器和双臂协调器仍是两个ROS节点，但共用一个进程；内部直接提交同一套
协调器事务，省去本机DDS服务往返。`/robot/arms/set_targets` 对外服务仍由同一个协调器提供。
采集期间树莓派将原始JPEG和状态动作顺序写入 `.runtime/recording-cache/episode-<ID>/frames.bin`，
同时保留配置、标定资产和缓存身份清单。录制期间不传输训练图像，不等待电脑磁盘或网络吞吐。
结束后传到电脑暂存，再并行编码四路H264 MP4，避免每帧等待编码挤占30FPS采样预算。
界面“保存中”包含传输、编码和校验，期间遥操继续，不能开始下一回合，等完成再按R；
最终数据集仍是LeRobot v3的Parquet和MP4，临时JPEG流在成功提交或人工丢弃后清理。
传输队列最多32个事件、32MiB，发送速度较慢时只增加回合结束后的等待，不影响已经完成的采样帧率。
完整样本带编号并由电脑确认，最终还核对样本数和包含全部JPEG、状态动作的SHA256；原始树莓派时间戳不改变。
电脑完成LeRobot提交并返回匹配回执后，树莓派才删除这一回合缓存。已保存的旧树莓派数据集不参与清理。
传输/电脑保存失败会保留树莓派缓存，按B可重试；重试通过缓存ID识别已提交回合，避免重复保存。
重启后使用同一数据集及任务配置并加`--resume`，会识别未完成缓存并先恢复传输；本地仅清理同一缓存ID的未提交暂存。
磁盘或编码故障保留已提交回合及待检查缓存，不停止遥操。树莓派剩余空间不足128MiB时停止继续缓存并提示。

| 采集按键 | 功能 |
| --- | --- |
| R | 开始第一回合、下一回合或重录；开始后需完整新帧才写入 |
| B | 提前结束并保存当前有效回合 |
| N | 丢弃当前未提交回合，再按R重录；不删除已保存回合 |
| Q | 停止底盘并退出电脑遥操，同时请求保存有效前段 |

采集状态显示在原来的电脑终端；键盘焦点仍需在终端。
也可独立启动 `ros2 launch xyc_recording recording.launch.py`，复用已运行的相机和驱动话题；
正常遥操采集用上面的统一启动即可，不要重复启动采集器。单独启动数据源仍需电脑客户端接收才能保存数据。
采集仅订阅传感器/控制快照，不会为凑齐数据自行上力、转动或给升降回零。

默认 `--record-scope mobile` 保存 **18维**，顺序明确如下：

| 下标 | observation.state | action |
| --- | --- | --- |
| 0..6 | 左臂6关节＋夹爪，标定归一化单位 | 观测后本轮新生成、发送并被驱动接受的目标位置 |
| 7..13 | 右臂6关节＋夹爪，标定归一化单位 | 观测后本轮新生成、发送并被驱动接受的目标位置 |
| 14 | 升降高度，m | 升降三值命令-1/0/+1，对应-0.015/0/+0.015m/s |
| 15..17 | 底盘实测vx、vy、wz，m/s、m/s、rad/s | 三值命令-1/0/+1；平移固定±0.15m/s，旋转固定±0.60rad/s |

底盘实测速度由轮速经过配置的轮径/轮位换算；升降state使用回零后的高度。
动作按原项目的“读取观测→生成并发送本轮指令→保存”约定记录请求值：双臂使用遥操规划器发送给协调器的新目标，
底盘/升降使用本轮发送并通过驱动校验的新速度请求。物理执行时的编码器量化、行程边界和保护可能影响实际结果，
它们通过实测state另行反映；不把旧的保持目标重新打时间戳当新动作，也不把下一帧state平移来当动作。
仅采双臂可选 `--record-scope arms`，保存14维并保留四路图像，不能混入同一个18维数据集。

采集使用四相机各8帧缓存，挑选读取时间差不超过33.3ms、距当前不超过150ms的新图组，
每张图只使用一次。双臂状态匹配最近样本；取四幅图和状态中最晚的观测时间，
匹配此后100ms内首个新生成、已发送且被接受的双臂指令和底盘/升降指令。
两路执行接口分别按各自的新命令事件匹配，不声称18轴在硬件上原子下发。
命令携带编号及生成、发送、接受时间，只使用一次；即使数值不变，录制期间也按新的有效主臂输入提交同值的新指令，
电机驱动保留同值目标不重复写硬件的优化。仅遥操、不录制时保留原来的目标去重逻辑。
同一时钟下记录每个关节和相机的来源时间；这是软件收图配对，不代表硬件同时曝光。
`action.sample_times` 是新指令生成时间，`action.sent_times`、`action.accepted_times` 分别记录发送和接受时间；
`action.arm_command_id`、`action.mobile_command_id` 对应具体新命令，`meta/robot.json` 保存动作约定版本。
有效观测/动作中断超过0.5秒时结束当前回合，请求电脑保存有效前段；不阻塞或关闭遥操。
录制时树莓派顺序缓存；结束后的传输及电脑视频编码独立于电机控制，电机控制不等待电脑磁盘。
采集复用相机的JPEG压缩话题，质量由 `preview_quality` 控制，当前为75；原始图像话题不受影响。

保存目录在本地电脑的项目下 `data/datasets/xyc/pick_toys/`（或`--data-dir`指定的位置）：

```text
meta/
  info.json                     v3.0、FPS、特征名称和维度
  stats.json                    均值/方差及q01/q99等归一化统计
  tasks.parquet                 任务文字和编号
  episodes/chunk-000/file-000.parquet  回合索引、数据范围、四视频偏移和统计
  robot.json                    主从臂标定、单位和坐标约定
  quality/episode-000000.json    实际新帧率、时间差和控制会话
data/chunk-000/file-000.parquet  状态、动作、帧编号及真实来源时间
videos/observation.images.<相机>/chunk-000/file-000.mp4
```

每个已提交回合独立占用一个v3文件，视频时间偏移在回合索引中保存；文件数达到1000后换chunk。
视频使用标准H.264，数据集时间网格是 `frame_index/fps`；`observation.source_time` 等列保留实际时间差。
保存时生成真实视频帧数和数值统计，图像统计使用定间隔RGB采样，不依赖训练环境。
正常结束会封装视频并提交回合；突然断电/强杀留下 `.pending-*` 或 `.writer.lock` 时，
先保留并检查这些文件，采集器会拒绝盲目覆盖或继续追加。保存完成前不要关闭电脑采集程序或电脑电源。

仅开启采集模块、没有启动录制时，不生成训练样本。LeRobot v3格式可由官方读取器加载；
本项目的兼容测试会检查两回合、四路视频、RGB颜色、18维状态/动作及动作序列的回合边界。

## 开发与验证

硬件控制分工和接口见 `ros_ws/src/xyc_hardware/README.md`。
修改代码后在树莓派重建对应功能包；旧进程不会自动加载新代码，切换电机节点前先支撑双臂。
测试使用假的串口、电机和隔离 ROS 域，不应与实机节点共用测试域：

```bash
cd ~/pi05_from_zero/ros_ws
source install/local_setup.bash
ROS_DOMAIN_ID=181 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST XYC_ISOLATED_TEST=1 python -m pytest -q src/xyc_hardware/test src/xyc_observation/test src/xyc_camera/test/test_preview.py src/xyc_recording/test
```
