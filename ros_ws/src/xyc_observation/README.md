# xyc_observation：统一观测接口

订阅四路图像、双臂标定状态和底盘/升降实测状态，按时间戳近似配对，发布一条包含图像和 18 维状态的观测。该节点不打开串口、不发送电机命令。

## 启动

新终端先准备环境：

```bash
unset PYTHONPATH
conda activate xyc
source ~/pi05_from_zero/ros_ws/install/local_setup.bash
```

输入节点尚未启动时，一次启动四个相机节点、两个只读电机节点和一个观测节点：

```bash
ros2 launch xyc_observation observation.launch.py start_inputs:=true
```

已经启动了四路相机和两个带标定的状态节点时，只启动观测节点，避免重复占用设备：

```bash
ros2 launch xyc_observation observation.launch.py
```

`start_inputs` 默认 false。带输入启动时，电机节点使用 `calibrated:=true chassis_mode:=true`，只读反馈，不开启双臂运动控制。18维观测还要求 `/robot/chassis/state` 有有效升降高度，因此升降必须已经完成回零。已有状态节点若未启用标定或底盘模式，不会提供完整输入。如果已启动统一遥操，请保持 `start_inputs:=false`，避免重复打开电机串口。

另一个准备好环境的终端查看一条观测；`--no-arr` 避免打印图像字节：

```bash
ros2 topic echo /robot/observation --once --no-arr --qos-reliability best_effort
ros2 topic echo /robot/observation --once --field state --qos-reliability best_effort
```

没有观测输出时，查看每路输入和整体状态：

```bash
ros2 topic echo /robot/observation_status --once
```

## 消息约定

类型为 `xyc_interfaces/msg/Observation`，定义在同一工作区的 `xyc_interfaces` 包中。

| 字段 | 含义 |
| --- | --- |
| `header.stamp` | 本条观测的参考时间，取前方图像原始时间戳 |
| `published_at` | 观测节点检查并构造这条消息的时间 |
| `sequence` | 发布序号，从 0 递增，uint32 回绕 |
| `schema_version/schema_id` | 消息约定版本和配置内容的 SHA256 标识 |
| `camera_names/camera_topics/images` | 固定四路顺序及实际原始图像，保留每幅图像自己的 header |
| `state_names/state/state_units` | 固定顺序的 18 个状态名、实测状态和单位 |
| `state_stamps` | 与18维状态一一对应的采样时间 |
| `arm_batch_stamps/mobile_batch_stamp` | 左右臂及底盘/升降批次时间 |
| `calibration_ids` | 左、右臂标定文件内容的 SHA256，与状态生产者核对 |
| `max_time_skew_s` | 本条所有图像、批次及关节采样时间的最大差值 |
| `oldest_source_age_s` | 构造消息时最旧输入的年龄 |
| `valid` | 本节点只发布通过全部检查的消息，值为 true |

相机顺序固定为 `forward, chest, wrist_left, wrist_right`。

状态先左臂七维、再右臂七维，每臂顺序为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper`；最后四维固定为 `lift.height_m, base.vx_m_s, base.vy_m_s, base.wz_rad_s`。输入可以重排，本节点按名称恢复顺序。普通关节为标定范围 `[-100, 100]`，夹爪为 `[0, 100]`；升降为米，底盘为实测 m/s、m/s、rad/s。这里发布状态，不包含动作标签。

标定文件默认读取 `~/pi05_from_zero/config/calibration/{left,right}.json`，不修改文件。`schema_id` 根据图像格式、名称、关节顺序、状态单位、标定标识和部分时间参数生成。它用于识别数据约定，不是模型版本。

## 配对和有效性

`message_filters.ApproximateTimeSynchronizer` 对四路图像、两路臂状态和一路底盘/升降状态进行七路配对，默认时间窗口 40 ms、每路最多 8 条。节点默认和配置均为30 Hz，只取最新的一组配对，每组最多使用一次。原始相机和电机输入仍按自身频率运行。

输出前检查：四路身份/图像尺寸/编码/字节长度正确；每臂七个关节齐全且无重复；底盘四个状态名及单位正确；状态有限、升降在已验证0～0.6m行程内、没有驱动错误且不在回零；标定标识匹配；输入不超过250ms、不过度超前；包含单关节时间在内的全部时间差不超过50ms。底盘是否上力不是状态有效性的条件。

缺一路、数据无效或配对不合格时，不发布该条观测，不补零、不重复上一帧。`/robot/observation_status` 每秒发布标准 `DiagnosticArray`，给出缺失来源和拒绝原因。下游即使收到 `valid=true`，也必须检查消息年龄，停止时不能一直沿用最后一帧。

相机时间戳是树莓派读到图像后的软件时间，电机时间戳是串口读回时间。因此这里实现的是软件近似配对，不是相机曝光和电机采样的硬件同步。当前按同一台树莓派的 ROS 时钟运行；离线 bag 回放与 `/clock` 的统一启动不包含在本步骤。

## 配置和文件

- `config/observation.yaml`：输入名称、图像格式、关节顺序、30 Hz 目标输出和时间阈值。
- `launch/observation.launch.py`：启动观测节点，可选同时启动现有相机及电机节点。
- `xyc_observation/observation_node.py`：ROS 订阅、近似同步、发布与诊断。
- `xyc_observation/observation_builder.py`：格式、时间、关节排序及标定标识检查，无硬件访问。
- `test/test_observation.py`：消息序列化、坏数据、缺一路和恢复等测试，使用模拟数据，不控制实机。

参数在启动后只读。修改 YAML 后重新构建、加载工作区并重启节点。当前schema v2固定四路相机和18维状态。

四幅 640×480 BGR 图像约 3.7 MB，每秒 30 条的像素负载约 111 MB（十进制）。输出使用 best effort、深度 1，不保证录制端收到每条消息；不以此接口的序号推断严格等间隔数据。沿用相机包的 16 MiB FastDDS 配置。不要直接打印图像数组或长时间录制未压缩观测；正式示教数据和 LeRobot 格式导出是后续步骤。

## 历史验证（2026-11-06，旧14维schema v1）

- 新增 11 项测试全部通过：包括原始像素及消息序列化、固定关节顺序、时间与标定校验，以及真实 ROS 传输下缺相机停止输出、恢复后继续。原有硬件包 67 项模拟测试全部通过，没有进行实机运动测试。
- 树莓派实机以 `start_inputs:=true` 启动 7 个业务节点，只读取相机和电机状态。在约 22 秒的启动及检查窗口内收到 167 条完整观测；收到数据后的频率约 10.001 Hz，参考图像时间频率约 9.995 Hz。
- 本次样本最大时间差 34.934 ms，最旧源数据年龄最大 59.397 ms，四幅图像各为 640×480 BGR、921600 字节，状态均为 14 维。最后诊断全部 OK，拒绝数 0。
- 两份标定文件的内容散列在检查前后一致；测试结束后 7 个业务节点均正常退出。上述是本次短时实测结果，不表示长期运行或跨网络传输已经验收。

同日将配置文件 `publish_hz` 调到 30.0 Hz，并调整相机节点读取节奏：读取定时器按配置 FPS 的两倍触发，V4L2 `read()` 等待实际下一帧，从而减少定时器与相机采集时钟错过。相机硬件请求帧率仍是 30 Hz，分辨率仍是 640×480。普通 ROS 相机话题实际频率取决于相机和系统负载。

树莓派四路相机与双臂同时运行约 125 秒（含启动），后约 119 秒的有效采样收到了 3462 条观测，平均 29.083 Hz；最大接收间隔 69.6 ms，没有大于 100 ms 的间隔；最大来源时间差 40.5 ms，最大来源年龄 61.8 ms。观测值及四路图像检查全部通过，最后的整体诊断为 OK。测试接收端在有效采样期间未发现观测序号跳号；同步前的输入与匹配输出并非完全一一对应，不能据此声称源数据零丢帧。

另一次启动时左腕相机发生 USB 断开后重新连接，观测节点因缺一路而停止输出；重新启动后完成上述测试。USB 断开的原因尚未确定。125 秒的结果不能保证长时间运行、网络传输或以后录制写盘都保持该频率。

18维schema v2新增了带时间戳的 `/robot/chassis/state` 和七路同步；本地编译及模拟测试用于检查消息与严格校验，仍需在树莓派重新构建后完成实机频率和时间差验证。
