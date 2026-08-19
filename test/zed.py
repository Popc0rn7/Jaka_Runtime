import os
import sys
import hydra
from omegaconf import DictConfig
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.zed import ZedCamera


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig):
    # 初始化 Zed Camera
    zed_camera = ZedCamera(**cfg.zed)
    zed_camera.start()
    time.sleep(2)

    for i in range(10):
        start = time.perf_counter()
        agent_view_image = zed_camera.read()
        end = time.perf_counter()

        print(f"Cost time: {end - start:.4f}s")
    zed_camera.stop()


if __name__ == "__main__":
    main()
