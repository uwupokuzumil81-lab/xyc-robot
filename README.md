# XYC Robot

一个连接双臂、四路相机、底盘和升降机构的机器人平台。电脑负责主臂读取、键盘、数据保存和模型运行，树莓派负责 ROS 2 Jazzy 与实机驱动。支持 Windows / Linux 电脑，训练可移到云 GPU。

遥操、数据回放和模型推理共用双臂协调器、电机节点、速度参数与保护逻辑。当前已接入 ACT；Π0.5 模型本体尚未接入。

## 代码结构

```text
src/xyc/
  contracts.py             相机、状态、动作顺序、单位与固定速度
  robot/
    profile.yaml           唯一默认机器人配置
    profile.py             配置加载、校验与路径解析
    feetech.py              无 ROS 依赖的串口读写和报文解析
    calibration.py         标定坐标、记录、检查与维护工具
  transport/               遥操协议、运动协议、后台独占锁
  teleop/
    leader.py              主臂发现、标定读取和单帧采样
    input.py               共用主臂采样线程、暂停和最新帧
    keyboard.py            共用键盘状态机、长按及组合方向
    mapping.py             主从臂目标映射
    cli.py / calibrate.py   已安装的桌面命令与维护工具
  observation.py           相机与状态的统一时间匹配
  recording/               回合缓存、传输和 LeRobot v3 保存
  learning/                数据读取、模型注册、训练、checkpoint和入口
  models/act/              ACT 模型、配置和适配器
  runtime/                 推理调度、人工准备、执行与评估记录
  images.py                原始 RGB 解码和显式图像变换工具
  assets.py                标定版本计算
  io.py                    原子 JSON 保存、内容哈希
  paths.py / terminal.py / locking.py
ros_ws/src/
  xyc_interfaces/          ROS 消息和服务
  xyc_camera/              图像采集节点、浏览器预览
  xyc_hardware/            电机节点、控制协调、底盘/升降与保护
  xyc_observation/         统一观测的 ROS 输出
  xyc_recording/           ROS 录制节点
config/
  calibration/            从臂标定资产
  leader_calibration/     主臂标定资产
  train/act.yaml           ACT 训练配置
tools/                     三个薄脚本入口，共用已安装的实现
```

公共 Python 包不导入 ROS 功能包，也不通过修改搜索路径调用 ROS 源码。ROS 节点引用公共包。
`ros_ws/build`、`install`、`log`、`.setup`、数据集和权重属于本机产物，不作为源码覆盖另一台机器。

## 机器人配置

安装后可直接使用 `xyc-teleop`、`xyc-leader` 和 `xyc-learning`；`tools/` 中的脚本调用相同实现。

默认配置是 [`src/xyc/robot/profile.yaml`](src/xyc/robot/profile.yaml)，随 Python 包安装。主臂 USB 序列号、从臂串口、舵机身份、四路相机、机械尺寸、SSH 默认地址、端口及树莓派项目目录都从这里读取。

更换机器人时可复制一份配置，并在相应电脑与树莓派设置：

```bash
export XYC_ROBOT_CONFIG=/绝对路径/robot.yaml
```

`XYC_ROBOT_HOST`、`XYC_ROBOT_USER` 可覆盖配置中的 SSH 主机和用户。`XYC_PROJECT_ROOT` 指定电脑端源码/资产项目根目录。地址改变不需要修改 Python 代码。配置在启动时读取，修改后重启相应进程。当前采集要求四台相机使用相同的偶数分辨率；保存器尺寸同样从配置读取。

默认通道为：17882 遥操、17883 图像及数据传输、17886 回放/模型控制；17884 是树莓派内部的数据缓存服务。手动 SSH 和客户端必须使用相同配置端口。

## 安装与构建

电脑在项目根目录执行，不需要 ROS：

```bash
python -m pip install -r requirements.txt
```

模型环境额外安装 GPU 对应的 PyTorch，再安装 `python -m pip install -e '.[act]'`。依赖版本统一声明在 `pyproject.toml`；ROS 环境和训练环境分别创建，避免替换 ROS 的 OpenCV 依赖。

树莓派使用 `environment.yml` 创建 `xyc` 环境后，在完整项目目录运行：

```bash
conda activate xyc
python -m pip install --no-deps -e .
cd ros_ws
colcon build --symlink-install
source install/local_setup.bash
```

安装和当前 Linux 连接示例见 [docs/linux.md](docs/linux.md)。代码更新后需重启后端；不同版本的电脑客户端和机器人后端应先同步再联机。

## 遥操与录制

电脑终端一建立控制转发并启动树莓派后端。以下使用已配置的 SSH 别名 `xyc-robot` 和默认端口、目录：

```bash
ssh -tt -o ExitOnForwardFailure=yes -L 17882:127.0.0.1:17882 xyc-robot "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate xyc && source ~/pi05_from_zero/ros_ws/install/local_setup.bash && ros2 launch xyc_hardware teleop.launch.py side:=both live:=true with_motors:=true with_cameras:=true follow_mode:=absolute chassis_mode:=true recording:=true'"
```

电脑终端二进入项目根目录。不录制时运行：

```bash
python tools/teleop_stream.py leader --side both --keyboard
```

录制一回合时运行：

```bash
python tools/teleop_stream.py leader --side both --keyboard --record --record-scope mobile --dataset xyc/pick_toys --task "Pick up the toy and put it in the box." --fps 30 --episode-frames 900 --episodes 1
```

默认保存到电脑 `data/datasets/`，可用 `--data-dir` 指定目录；同名数据集继续追加时加 `--resume`。
录制时先在树莓派顺序缓存，回合结束后传到电脑、写入和校验，再允许开下一回合。保存确认前保留缓存；中断重传不会覆盖其他已提交回合。首次创建的数据集元数据在完整写入后一次发布，初始化失败可以重试。

| 按键 | 功能 |
| --- | --- |
| W/S、A/D、J/L | 前后、左右平移、旋转，允许组合方向 |
| I/K | 升降；使用前需要建立零点 |
| H | 下行回零；录制中需先结束当前回合 |
| 空格 | 停车 |
| R、B、N | 开始、保存、丢弃当前录制回合 |
| Q | 停车并退出电脑客户端 |

大小写均可。Windows 使用按键按下/松开状态；Linux 终端使用按键重复，停止重复后约 0.25 秒停车。
新的换向或停车按键优先于长按续发；松开当前方向键后该轴停车。遥操和模型推理前的人工准备界面使用同一键盘状态机与主臂采样器。
遥操报停时，电机节点调用公共停车逻辑。底盘和升降写入零速并取消回零，双臂关闭已上力关节的扭矩，停止追踪上一目标。故障锁定拒绝新的运动和使能请求，重新连接电脑不会清除锁定。排除故障后重启遥操后端。

客户端自动建立独立数据 SSH 通道并打开四路相机网页。浏览器只显示图像；控制时将焦点切回终端。
相机 `fps`、`jpeg_quality` 决定采集和推理输入，网页显示频率单独由配置的 `preview_fps` 控制；调低网页频率不会降低采集流频率。

## 数据与统一动作

`state` 为双臂14维标定位置、升降高度 m、底盘实测 vx/vy/wz；`action` 为双臂14维目标及四个 `-1/0/+1` 命令。
固定轴速度为升降 0.015 m/s、平移 0.15 m/s、旋转 0.60 rad/s。组合动作超出轮速上限时，同比降低机体速度。
遥操客户端也按这四个三值发送移动命令，由机器人端换算速度。使用实机遥操前，电脑与树莓派须运行同一版本。

四路 USB 图像按读取完成时间与状态配对，记录真实来源时间；这不等同于硬件同时曝光。标签使用观测之后新生成、实际发送并被驱动接受的命令。

```text
数据集/
  meta/info.json                  FPS、名称、维度与 v3 格式
  meta/robot.json                 标定资产和单位
  meta/stats.json                 数值与图像统计
  meta/tasks.parquet              任务编号和文字
  meta/episodes/chunk-*/file-*.parquet
  meta/quality/episode-*.json      新帧率、时间差与中断原因
  data/chunk-*/file-*.parquet      state、action、编号与来源时间
  videos/observation.images.<相机>/chunk-*/file-*.mp4
```

采集、ROS 观测和实时推理共用 `ObservationBuffer`。数据读取器输出原始 RGB；各模型适配器负责自己的缩放、归一化、文本处理和反归一化。

## 模型与回放

完整命令见 [docs/learning.md](docs/learning.md)。

```bash
python tools/learning.py train --config config/train/act.yaml
python tools/learning.py infer --checkpoint 权重目录 --dataset 数据集目录 --episode 0 --output /tmp/prediction.json
python tools/learning.py deploy --checkpoint 权重目录 --device cuda --prepare-mode keyboard
```

推理先进入人工准备界面，调整后按 P 开始；`--prepare-mode teleop` 使用主臂加键盘，否则只用键盘调整底盘/升降。P 暂停，Y/N 标成功或失败并返回准备，Q 退出。

实机回放需要先退出遥操后端，再运行 `replay.launch.py` 并转发17886，然后在电脑执行：

```bash
python tools/learning.py replay --dataset 数据集目录 --episode 0 --speed 1 --execute --home-lift
```

不加 `--execute` 只检查数据。`--home-lift` 明确允许需要时下行回零；回放会先对齐录制首帧。推理开始时使用人工调整后的场景和观测。

## 控制与维护

启动上力先读取当前位置并确认，再保持当前位置；持续目标每条总线批量写入。正常到位继续上力，位置偏移报警时保留目标。夹爪接触采用限扭矩夹持。电流、温度、反馈中断、通信结果不明和退出时的处理共用同一套底层实现。
底盘和升降共用 `/robot/mobile/command`，三值目标和绝对执行期限一次发送；旧的两个速度话题入口已移除。源码更新后需同步并重新构建机器人端的 `xyc_interfaces` 与后端，控制协议为 v3。
电脑客户端退出后，双臂可能继续保持；电机后端关闭会尝试脱力。重启后端前先支撑好双臂。

```bash
python tools/leader_arm.py left calibrate
python tools/leader_arm.py right calibrate
ros2 run xyc_hardware calibrate_arm --arm left --check
```

标定是独占串口的维护工具，不与运动后端同时运行。硬件接口和保护说明见 [xyc_hardware/README.md](ros_ws/src/xyc_hardware/README.md)。

## 开发验证

```bash
# 训练环境；使用临时数据和模拟输入，不打开设备
python -m pytest tests/learning -q
# ROS 环境加载工作空间后
python -m pytest ros_ws/src -q
```

ROS 联调使用模拟驱动及隔离域：双臂协调器、底盘/升降和相机为181，回放为183，录制为184；设置 `XYC_ISOLATED_TEST=1` 和 `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`，分别执行对应测试文件。测试域不与实机节点混用。
