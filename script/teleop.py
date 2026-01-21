import os
import sys
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
    robot = JakaS5(ip="192.168.2.121")
    robot.start()

    # 初始化 Ultrahands Client
    ultrahands = UltrahandsClient(**cfg.client)
    ultrahands.start()

    # 主循环
    dt = 1.0 / 30.0
    next_tick = time.perf_counter()
    while True:
        next_tick += dt
        angles = ultrahands.input_report.angles
        if angles is not None:
            joint_pos = angles[:7]
            robot.JointCtrl(joint_pos)

        # 频率控制
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

if __name__ == "__main__":
    main()
