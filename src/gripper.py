"""DH AG95 position conversion helpers."""


def normalized_to_position(target: float, position_scale: int) -> int:
    """Convert a normalized target to a device position using its scale."""
    target = float(target)
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"Gripper target must be in [0, 1], got {target}")
    if position_scale <= 0:
        raise ValueError("Gripper position_scale must be positive")
    return round(target * position_scale)
