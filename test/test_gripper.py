import pytest

from src.gripper import normalized_to_position


@pytest.mark.parametrize(
    ("target", "position_scale", "expected"),
    [
        (0.0, 1000, 0),
        (1.0, 1000, 1000),
        (0.25, 1000, 250),
        (0.25, 800, 200),
    ],
)
def test_normalized_to_position_uses_a_device_scale(
    target: float, position_scale: int, expected: int
) -> None:
    assert normalized_to_position(target, position_scale) == expected


def test_normalized_to_position_rejects_out_of_range_targets() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        normalized_to_position(1.01, 1000)


def test_normalized_to_position_rejects_a_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="position_scale must be positive"):
        normalized_to_position(0.5, 0)
