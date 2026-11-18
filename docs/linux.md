# Linux 电脑使用

Ubuntu 24.04 已验证。本机项目位于 `~/Projects/xyc-robot`，旧的中文目录保留符号链接。
ROS 接口生成工具会错误解析中文构建路径，因此使用英文项目路径构建。
树莓派上的项目仍位于 `~/pi05_from_zero`。

## 环境

`xyc` 用于 ROS 2 Jazzy、主臂串口、遥操、采集和回放；`xyc-act` 用于 ACT 训练与推理。
两者分开安装，避免训练用的图像依赖覆盖 ROS 的 `cv_bridge` 依赖。

新电脑重建 ROS 环境：

```bash
cd ~/Projects/xyc-robot
conda env create -f environment.yml
conda activate xyc
python -m pip install -r requirements.txt -e '.[replay]'
cd ros_ws
colcon build --symlink-install
source install/local_setup.bash
```

`environment.yml` 限制 `setuptools<80`，保证 colcon 的符号链接构建可用。
串口用户需加入 `dialout` 组，执行后重新登录桌面生效：

```bash
sudo usermod -aG dialout "$USER"
```

本次已配置串口权限，无需重复标定。左右主臂按 USB 序列号识别，不依赖 `/dev/ttyACM0` 的编号。

## 本机连接

有线连接配置 `xyc-robot-wired`：电脑 `192.168.77.1/24`，树莓派 `192.168.77.2/24`。
该连接不接管互联网默认路由。SSH 配置 `~/.ssh/config` 中的 `xyc-robot` 指向这台树莓派，
并已配置密钥登录。本局域网有重名的 `raspberrypi.local`，日常使用已核对硬件身份的别名：

```bash
ssh xyc-robot
```

`xyc` 环境已设置 `XYC_ROBOT_HOST=xyc-robot`，电脑端独立数据 SSH 通道也使用这个别名。
换机器人时修改 SSH 主机配置，并核对串口身份与标定文件。

## 遥操与采集：两个终端

终端一在电脑运行，启动树莓派后端并转发控制端口；保持窗口开启：

```bash
ssh -tt -o ExitOnForwardFailure=yes -L 17882:127.0.0.1:17882 xyc-robot "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate xyc && source ~/pi05_from_zero/ros_ws/install/local_setup.bash && ros2 launch xyc_hardware teleop.launch.py side:=both live:=true with_motors:=true with_cameras:=true follow_mode:=absolute chassis_mode:=true recording:=true'"
```

终端二在电脑运行。不录数据时：

```bash
conda activate xyc
cd ~/Projects/xyc-robot
python tools/teleop_stream.py leader --side both --keyboard
```

录一条 900 帧数据时，改用下面的客户端命令。数据集名称需使用未存在的名称：

```bash
python tools/teleop_stream.py leader --side both --keyboard --record --record-scope mobile --dataset xyc/pick_toys_linux_01 --task "Pick up the toy and put it in the box" --fps 30 --episode-frames 900 --episodes 1
```

客户端自动建立独立的 17883 数据连接并打开四路相机网页，无需手动再转发 17883。
H 完成升降回零后按 R 开录；B 保存，N 丢弃，Q 停车并退出客户端。
Linux 终端用按键重复维持方向，停止重复后约 0.25 秒停车。
录制中树莓派缓存，结束后传到电脑 `data/datasets/`，完成校验后清理缓存。
退出客户端后双臂可能继续保持；退出终端一的电机后端会尝试脱力，应先支撑双臂。

## 回放：先退出遥操后端

终端一：

```bash
ssh -tt -o ExitOnForwardFailure=yes -L 17886:127.0.0.1:17886 xyc-robot "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate xyc && source ~/pi05_from_zero/ros_ws/install/local_setup.bash && ros2 launch xyc_hardware replay.launch.py'"
```

终端二：

```bash
conda activate xyc
cd ~/Projects/xyc-robot
python tools/learning.py replay --dataset data/datasets/xyc/pick_toys_linux_01 --episode 0 --speed 1 --execute --home-lift
```

不加 `--execute` 只检查数据，不连接机器人。`--frames 30` 可只回放前 30 帧。

## ACT

本机 `xyc-act` 已安装 PyTorch 2.7.1 / torchvision 0.22.1 的 CUDA 12.8 版本，支持 RTX 5060。
重新创建时先安装对应 GPU 的 PyTorch，再安装 `python -m pip install -e '.[act]'`。
项目已有训练参数保持不变；通过命令指定实际存在的数据集：

```bash
conda activate xyc-act
cd ~/Projects/xyc-robot
python tools/learning.py train --config config/train/act.yaml --dataset data/datasets/xyc/pick_toys_linux_01
```

当前 ACT 已验证训练、续训、权重恢复、离线推理和运行时接口。
环境短训产生的权重只用于验证程序，实机任务推理需要正式训练的 checkpoint。

## 公共配置与安装包

机器人配置统一在 `src/xyc/robot/profile.yaml`；使用其他配置时设置 `XYC_ROBOT_CONFIG`。主臂采样、键盘、协议和保存器已收进 `xyc` 公共包，电脑端无需添加 ROS 源码目录。模型适配器统一处理训练与实时输入的原始 RGB 图像。

更新公共包时分别在两个新终端激活 `xyc`、`xyc-act`，各自运行 `python -m pip install --no-deps --no-build-isolation -e .`。安装前清除上一环境残留的 `PYTHONPATH`，避免把另一个环境的安装当作当前环境卸载。ROS 环境安装完成后再加载工作空间。
