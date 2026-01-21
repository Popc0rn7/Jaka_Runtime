import ctypes
import os
from pathlib import Path
import sys
import time
import hydra

# set SDK env

sys_path = os.path.join(os.getcwd(), "SDK")
sys.path.append(sys_path)
sys_path = os.path.join(os.getcwd(), "SDK", "libjakaAPI.so")
ctypes.CDLL(sys_path)

# set cwd path
sys.path.append(str(Path(__file__).parent.parent))
print(sys.path)
from hardware.robot.robot import S5
from hardware.gripper.robotiq_gripper import RobotiqGripper
from omegaconf import DictConfig

# import ultrahands-piper modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.ultrahands import UltrahandsClient

OPEN = 1000
CLOSE = 0


@hydra.main(version_base=None, config_path="../config", config_name="teleop")
def main(cfg: DictConfig):
    # 启动robot
    robot = S5()
    robot.start()

    # 启动夹爪
    gripper = RobotiqGripper(port="/dev/ttyUSB0")
    gripper.init_gripper()
    # for test
    gripper.set_gripper(CLOSE)
    gripper.set_gripper(OPEN)
    # 初始化 Ultrahands Client
    ultrahands = UltrahandsClient(**cfg.client)
    ultrahands.start()

    # 时间参数
    period = 1.0 / cfg.teleop.hz
    next_time = time.perf_counter()

    print("等待手柄a键按下")
    while True:
        start = ultrahands.input_report.btn_a
        if not start:
            time.sleep(0.05)
        else:
            break
    print("开始遥操")

    print("等待手柄a键松开")
    while True:
        start = ultrahands.input_report.btn_a
        if not start:
            break

    while True:
        next_time += period
        ctl_data = ultrahands.input_report
        end = ctl_data.btn_a
        if end:
            break
        # robot control
        left_position = ultrahands.input_report.angles[:7]
        robot.JointCtrl(0, left_position)
        right_position = ultrahands.input_report.angles[7:14]
        robot.JointCtrl(1, right_position)

        # gripper control
        # 控制夹爪
        if ctl_data.btn_rb:
            gripper.set_gripper(CLOSE)
        elif ctl_data.btn_rt:
            gripper.set_gripper(OPEN)

        # 频率控制
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # 超时则立即进入下一次循环（可能丢帧）
            next_time = time.perf_counter()
    print("结束遥操")


if __name__ == "__main__":
    main()
