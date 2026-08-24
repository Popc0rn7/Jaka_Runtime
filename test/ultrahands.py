import os
import sys
import hydra
from omegaconf import DictConfig

# import modules from the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.ultrahands import *


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    client = UltrahandsClient(**cfg.client)
    client.start()

    client.input_report.print_static_labels()
    while True:
        client.input_report.print_status()


if __name__ == "__main__":
    main()
