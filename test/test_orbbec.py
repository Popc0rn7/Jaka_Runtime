from hardware.orbbec import OrbbecCamera


def test_orbbec_camera_accepts_a_device_selector() -> None:
    camera = OrbbecCamera(device="orbbec-001")

    assert camera.device == "orbbec-001"
