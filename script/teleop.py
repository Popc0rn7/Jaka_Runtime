import os
import sys
import csv
import time
import hydra
from omegaconf import DictConfig

# import modules from the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.jaka_s5 import JakaS5
from hardware.ultrahands import UltrahandsClient


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig):
    # 启动robot
    robot = JakaS5(ip="192.168.2.121", freq_hz=30)
    robot.start()

    # 初始化 Ultrahands Client
    ultrahands = UltrahandsClient(**cfg.client)
    ultrahands.start()

    # main loop
    while True:
        ramp_to_ultrahands(robot, ultrahands)  # 缓慢移动到 Ultrahands 位置
        teleop(robot, ultrahands)  # 进入遥操作循环


def ramp_to_ultrahands(robot: JakaS5, ultrahands: UltrahandsClient):
    print("press X to start ramping to ultrahands position...")
    last_x = False
    while True:
        x_pressed = bool(ultrahands.input_report.x)
        if x_pressed and not last_x:
            break
        last_x = x_pressed
        time.sleep(0.01)

    print("arm will move to ultrahands position in 2 seconds...")
    print("don't move ultrahands during this period.")
    time.sleep(1.0)
    angles = ultrahands.input_report.angles
    robot.JointCtrl(angles[:7], step_num=250)  # 2 seconds
    print("done.")


def record_angles(path: str, angles):
    if angles is None:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp"] + [f"joint_{i}" for i in range(len(angles))])
        writer.writerow([time.time()] + list(angles))


def teleop(robot: JakaS5, ultrahands: UltrahandsClient):
    print("teleop started, press Y to stop.")

    # record config
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    record_path = os.path.join(root_dir, "logs", f"angles_{timestamp}.csv")

    # frequency config
    dt = 1.0 / 30.0
    next_tick = time.perf_counter()

    # teleop loop
    last_y = False
    while True:
        next_tick += dt
        report = ultrahands.input_report

        # robot control
        angles = report.angles
        if angles is not None:
            record_angles(record_path, angles)
            joint_pos = angles[:7]
            robot.JointCtrl(joint_pos, 2)

        # check stop condition
        y_pressed = bool(report.y)
        if y_pressed and not last_y:
            print("Y pressed, stopping teleop.")
            break
        last_y = y_pressed

        # frequency control
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
