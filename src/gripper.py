"""DH AG95 position conversion helpers."""


def normalized_to_position(
    target: float, position_min: int, position_max: int
) -> int:
    """Convert a normalized target to a configured AG95 device position."""
    target = float(target)
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"Gripper target must be in [0, 1], got {target}")
    if position_min > position_max:
        raise ValueError("Gripper position_min must not exceed position_max")
    return round(position_min + target * (position_max - position_min))
