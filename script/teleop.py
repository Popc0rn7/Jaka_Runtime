import os
import sys
import time
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from PIL import Image
from pyDHgripper import AG95

# import modules from the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.jaka_s5 import JOINT_COUNT, JakaS5
from hardware.ultrahands import UltrahandsClient
from hardware.zed import ZedCamera
from src.data_collector import LeRobotDataCollector


def load_static_rgb_image(path: Path, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize to camera shape."""
    if not path.is_file():
        raise FileNotFoundError(f"Static camera image not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        target_ratio = width / height
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = round(source_height * target_ratio)
            left = (source_width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, source_height))
        elif source_ratio < target_ratio:
            crop_height = round(source_width / target_ratio)
            top = (source_height - crop_height) // 2
            image = image.crop((0, top, source_width, top + crop_height))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig):
    image_shape = (int(cfg.zed.output_width), int(cfg.zed.output_height), 3)  # HWC

    # 启动arm
    arm = JakaS5(ip="192.168.2.121", freq_hz=30)
    arm.start()

    # 启动gripper
    gripper = AG95(port=cfg.gripper.port)
    gripper.set_force(cfg.gripper.force)
    gripper.set_vel(cfg.gripper.velocity)

    # 初始化 Ultrahands Client
    ultrahands = UltrahandsClient(**cfg.client)
    ultrahands.start()

    # 初始化 Zed Camera
    zed_camera = ZedCamera(**cfg.zed)
    zed_camera.start()

    # 初始化数据采集器。一个运行目录可保存多个 teleop episode。
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    data_path = Path(root_dir) / "data" / "demo" / timestamp
    collector = LeRobotDataCollector(
        repo_id="local/jaka_s5_demo",
        root=data_path,
        fps=30,
        state_names=[*(f"joint_{i}.pos" for i in range(6)), "gripper.target_pos"],
        action_names=[*(f"joint_{i}.pos" for i in range(6)), "gripper.target_pos"],
        task="Just a Demo",
        camera_shapes={"wrist": image_shape, "agent_view": image_shape},
    )

    # Each press of the teleop stop control saves one LeRobot episode.
    while True:
        gripper.set_pos(0)  # 夹爪初始状态为闭合
        ramp_to_ultrahands(arm, ultrahands)  # 缓慢移动到 Ultrahands 位置
        try:
            teleop(arm, gripper, ultrahands, zed_camera, collector, image_shape)
        except KeyboardInterrupt:
            print("\nteleop interrupted.")
        finally:
            # Never leave an incomplete episode in the LeRobot dataset.
            if collector.dataset.has_pending_frames():
                collector.discard_episode()
            collector.finalize()
            ultrahands.stop()
            arm.stop()
            zed_camera.stop()
            break


def ramp_to_ultrahands(arm: JakaS5, ultrahands: UltrahandsClient):
    print(
        "press X to start ramping to ultrahands position in 2 seconds...",
        end="",
        flush=True,
    )
    last_x = False
    while True:
        x_pressed = bool(ultrahands.input_report.stick_l1_vertical)
        if x_pressed and not last_x:
            break
        last_x = x_pressed
        time.sleep(0.01)

    time.sleep(1.0)
    joint_pos = ultrahands.input_report.angles
    if joint_pos is None or len(joint_pos) < JOINT_COUNT:
        raise RuntimeError("No Ultrahands joint angles received for ramping")
    arm.JointCtrl(joint_pos[:JOINT_COUNT], step_num=250)  # 2 seconds
    print("done.")


def teleop(
    arm: JakaS5,
    gripper: AG95,
    ultrahands: UltrahandsClient,
    zed_camera: ZedCamera,
    collector: LeRobotDataCollector,
    image_shape: tuple[int, int, int],
):
    print("teleop started, press Y to stop...", end="", flush=True)

    static_image = load_static_rgb_image(
        Path(root_dir) / "data" / "dog.jpg", image_shape[0], image_shape[1]
    )

    # frequency config
    dt = 1.0 / 30.0
    next_tick = time.perf_counter()

    # gripper state
    gripper_open = False
    gripper_target = 0.0

    # teleop loop
    last_y = False
    last_rb = False
    while True:
        next_tick += dt
        report = ultrahands.input_report

        # arm control
        angles = report.angles
        if angles is None or len(angles) < JOINT_COUNT:
            raise RuntimeError("No Ultrahands joint angles received")
        arm.JointCtrl(angles[:JOINT_COUNT], 2)

        # gripper control
        rb_pressed = bool(report.stick_l2_vertical)
        if rb_pressed and not last_rb:
            gripper_open = not gripper_open
            gripper_target = 1.0 if gripper_open else 0.0
            gripper.set_pos(int(gripper_target * 1000))
        last_rb = rb_pressed

        # capture images
        agent_view_image = zed_camera.read()

        # collect data
        joint_state = arm.get_joint_position()
        # gripper_state = gripper.read_pos() / 1000.0  # sleep(0.08), may block
        gripper_state = gripper_target
        if joint_state is None or len(joint_state) < JOINT_COUNT:
            raise RuntimeError("No JAKA joint feedback received")
        collector.record_step(
            state=[
                *joint_state[:JOINT_COUNT],
                gripper_state,
            ],
            action=[
                *angles[:JOINT_COUNT],
                gripper_target,
            ],
            images={"wrist": static_image, "agent_view": agent_view_image},
        )

        # check stop condition
        y_pressed = bool(report.stick_l1_horizontal)
        if y_pressed and not last_y:
            collector.save_episode()
            print("done.")
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
