"""Dependency-isolated adapter for a remote OpenPI policy server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class OpenPIPolicyClient:
    """Request action chunks from an OpenPI WebSocket policy server.

    Importing ``openpi_client`` is intentionally delayed so the rest of the
    robot code can still be imported on machines that only collect data.
    """

    def __init__(self, host: str, port: int, *, action_key: str = "actions") -> None:
        if not host:
            raise ValueError("OpenPI policy host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError(f"OpenPI policy port must be in [1, 65535], got {port}")
        if not action_key:
            raise ValueError("OpenPI action key must not be empty")
        try:
            from openpi_client import image_tools
            from openpi_client import websocket_client_policy
        except ImportError as error:
            raise RuntimeError(
                "openpi-client is required for policy inference. Run `uv sync`."
            ) from error

        self._image_tools = image_tools
        self._action_key = action_key
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
        )

    def infer(self, observation: Mapping[str, Any], *, action_dim: int | None = None) -> np.ndarray:
        """Return a finite, rank-2 float32 action chunk.

        The action dimension is deliberately validated by the robot script,
        because it is specific to the robot/checkpoint pairing.
        """
        if not observation:
            raise ValueError("OpenPI observation must not be empty")
        result = self._client.infer(dict(observation))
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "OpenPI server response must be a mapping, "
                f"got {type(result).__name__}"
            )
        if self._action_key not in result:
            raise RuntimeError(
                f"OpenPI server response did not contain {self._action_key!r}"
            )
        actions = np.asarray(result[self._action_key], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(
                "OpenPI actions must have shape (action_horizon, action_dim), "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("OpenPI actions contain NaN or infinity")
        if action_dim is not None and actions.shape[1] != action_dim:
            raise ValueError(
                f"OpenPI actions must have {action_dim} values per action, "
                f"got shape {actions.shape}"
            )
        return actions

    def prepare_image(self, image: np.ndarray, size: int) -> np.ndarray:
        """Resize/pad an RGB image using OpenPI's client-side preprocessing."""
        if size <= 0:
            raise ValueError(f"Image size must be positive, got {size}")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
        prepared = self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(image, size, size)
        )
        prepared = np.asarray(prepared, dtype=np.uint8)
        if prepared.shape != (size, size, 3):
            raise RuntimeError(
                "OpenPI image preprocessing returned an unexpected shape: "
                f"{prepared.shape}"
            )
        return np.ascontiguousarray(prepared)

    def reset(self) -> None:
        """Reset policy episode state when supported by the server/client."""
        reset = getattr(self._client, "reset", None)
        if reset is None:
            return
        reset()
