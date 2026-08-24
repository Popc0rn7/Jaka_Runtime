# Configuration layout redesign

## Goal

Replace the single, mixed `config/ultrahands.yaml` with small Hydra YAML files
at the same `config/` directory level. The layout must cover the JAKA S5 arm
and DH gripper and make every script read device settings from configuration.

## Configuration files

`config/config.yaml` is the sole Hydra composition entry point. Its `defaults`
list composes the following same-level files:

- `jaka_s5.yaml`: JAKA S5 connection (`ip`, `freq_hz`), initial pose, and
  inference safety limits.
- `dh_gripper.yaml`: DH AG95 connection and operation parameters (`port`,
  `force`, `velocity`, normalized position range).
- `ultrahands.yaml`: Ultrahands client connection and protocol settings.
- `zed.yaml`, `orbbec.yaml`, `policy.yaml`, `display.yaml`, and `replay.yaml`:
  their existing focused settings.

The composed keys retain their existing logical names (`cfg.zed`,
`cfg.orbbec`, and so on). The robot moves from `cfg.robot` to
`cfg.jaka_s5`; the gripper remains `cfg.dh_gripper`.

## Script behavior

`script/teleop.py`, `script/inference.py`, and `script/replay.py` load
`config/config.yaml`. All JAKA S5 and AG95 construction values come from
`cfg.jaka_s5` and `cfg.dh_gripper`; existing hard-coded addresses and control
frequencies are removed. The operational behavior otherwise remains unchanged.

Hydra overrides continue to use the composed key, for example
`jaka_s5.ip=192.168.2.116` and `dh_gripper.port=/dev/ttyUSB0`.

## Validation

Add a configuration-composition test that loads `config/config.yaml` and
checks the expected component sections. Update unit tests for inference to use
the renamed JAKA configuration node, then run the Python test suite without
connecting to hardware.
