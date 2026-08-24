import pytest

from src.gripper import normalized_to_position


@pytest.mark.parametrize(
    ("target", "position_min", "position_max", "expected"),
    [
        (0.0, 0, 1000, 0),
        (1.0, 0, 1000, 1000),
        (0.25, 100, 900, 300),
    ],
)
def test_normalized_to_position_uses_configured_device_bounds(
    target: float, position_min: int, position_max: int, expected: int
) -> None:
    assert normalized_to_position(target, position_min, position_max) == expected


def test_normalized_to_position_rejects_out_of_range_targets() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        normalized_to_position(1.01, 100, 900)
