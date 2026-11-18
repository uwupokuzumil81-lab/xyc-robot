# xyc_observation：统一观测的 ROS 输出

采集、模型实时推理和此节点共用 `src/xyc/observation.py` 的 `ObservationBuffer`。
四路 JPEG、18 维状态、时间配对、有效性、标定版本与会话检查来自同一份实现。
本包只把配对结果转换成 `xyc_interfaces/msg/Observation` 并发布诊断，不控制电机。

先启动正常的遥操或模型后端，完成双臂上力和升降回零，再在树莓派启动：

```bash
conda activate xyc
source ~/pi05_from_zero/ros_ws/install/local_setup.bash
ros2 launch xyc_observation observation.launch.py
```

该入口复用现有输入，不再单独启动另一套相机或电机。
输入为四路 `/camera/<名称>/image_raw/compressed` 和两路
`/motors/<left或right>/recording_sample`。左侧样本同时包含底盘及升降状态。
没有完整有效控制状态时，诊断会说明原因，不发布占位观测。

```bash
ros2 topic echo /robot/observation --once --no-arr --qos-reliability best_effort
ros2 topic echo /robot/observation_status --once
```

当前 ROS schema 为 v3：图像由共用的 JPEG 解码，保持配置尺寸；
消息时间为四路相机软件读取时间的中位数。图像时间差不超过 33.3 ms，
状态选择距离相机参考时间不超过 80 ms，单关节反馈距离不超过 100 ms。
这些阈值直接来自公共观测模块；这里不再维护另一套近似同步参数。
此流程提供软件时间配对，不代表相机硬件同步曝光。

`config/observation.yaml` 仅配置尺寸、RGB/BGR 输出和 1～30 Hz 输出频率。
完整状态顺序、单位、相机顺序来自公共契约。消息携带源时间、两臂标定版本和 schema ID。
输出采用 best effort；四幅未压缩图像的数据量较大，电脑采集和推理仍使用同一公共模块的
JPEG 传输路径，无需绕行这个大消息。下游必须检查消息时间，避免复用旧图像。
