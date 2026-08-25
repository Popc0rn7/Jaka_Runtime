# 使用 UltraHands 遥操作 JAKA_S5 进行数据采集与 Policy 部署

## 1. Quick Start

环境用uv管理，拉取仓库并安装依赖：

```bash
git clone --recursive git@github.com:Popc0rn7/Jaka_Runtime.git
uv sync
```

对照`config/`下的配置文件修正当前的遥操作配置

## 2. Teleop

需要配合 [Ultrahands](hardware/ultrahands/README.md) 使用，配置一套组装完整的UltraHands并启动经过校对的 server，用以下指令检查是否联通：

```bash
uv run python -m hardware.ultrahands.report
```

然后可以放心启动 `teleop.py` 进行遥操作：

```bash
uv run script/teleop.py
```

首先会逐个启动各个硬件，对有问题的硬件进行排查，等到你看到`Press X to ...`的提示后说明已经进入遥操作待定状态（机器人还是被锁定），然后请将 Ultrahands 放到一个相对安全的位置，按下 X 键，为保护机器人，程序在实际遥操作之前会缓慢移动到该位置再解锁机器人开始遥操作。遥操作开始后自动开始采集数据并保存为`Lerobot V2.1`，而遥操作过程中可以按下 Y 键来停止这一轮遥操作，这样又会进入待定状态且保存前面一个episode的数据，之后可以继续按下 X 键开始下一轮遥操作。

## 3. Replay

提供了一个简单的 replay 脚本可以重放采集到的数据，使用方法如下：

```bash
uv run script/replay.py --dataset_root=<data_dir> --episode=<episode>
```

## 4. Policy Inference

支持了基础的 Policy Inference，当前只支持 Pi05，使用方法如下：

```bash
uv run script/inference.py
```

Press Enter 后会从当前位置开始执行 action。