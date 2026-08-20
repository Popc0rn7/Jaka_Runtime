import os
import sys
import time

import hydra
from omegaconf import DictConfig

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from hardware.orbbec import OrbbecCamera


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig):
    camera = OrbbecCamera(**cfg.orbbec)
    camera.start()
    try:
        time.sleep(2)
        for _ in range(10):
            start = time.perf_counter()
            image = camera.read()
            elapsed = time.perf_counter() - start
            print(f"shape={image.shape}, read_time={elapsed:.4f}s")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
