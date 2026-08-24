from pathlib import Path

from hydra import compose, initialize_config_dir
import yaml


def test_config_composes_hardware_sections() -> None:
    config_dir = Path(__file__).parents[1] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    assert cfg.jaka_s5.ip == "192.168.2.116"
    assert cfg.jaka_s5.freq_hz == 30
    assert cfg.dh_gripper.port == "/dev/ttyUSB0"
    assert cfg.client.host == "192.168.2.108"
    assert cfg.safety.max_joint_speed == 3.0
    assert set(cfg) >= {
        "jaka_s5",
        "dh_gripper",
        "client",
        "policy",
        "safety",
        "zed",
        "orbbec",
        "display",
        "replay",
    }


def test_config_accepts_device_overrides() -> None:
    config_dir = Path(__file__).parents[1] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "jaka_s5.ip=192.168.2.200",
                "dh_gripper.port=/dev/ttyUSB9",
            ],
        )
    assert cfg.jaka_s5.ip == "192.168.2.200"
    assert cfg.dh_gripper.port == "/dev/ttyUSB9"


def test_config_keeps_script_settings_in_the_composition_entry_point() -> None:
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["defaults"] == [
        "jaka_s5",
        "dh_gripper",
        "ultrahands",
        "zed",
        "orbbec",
        "_self_",
    ]
    assert set(config) >= {"policy", "safety", "display", "replay"}


def test_policy_config_contains_only_deployment_values() -> None:
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert set(config["policy"]) == {"host", "port", "task", "image_size"}
    assert config["safety"]["initial_ramp_steps"] == 250
