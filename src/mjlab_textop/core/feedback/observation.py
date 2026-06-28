from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Protocol

from mjlab_textop.core.feedback.image import get_observation_image_store


class TextOpObservationPublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None:
        """Publish one MJLab observation payload."""


@dataclass(frozen=True)
class UdpObservationPublisherCfg:
    host: str = "127.0.0.1"
    port: int = 8766
    image_store_key: str | None = None


class UdpObservationPublisher:
    def __init__(self, cfg: UdpObservationPublisherCfg) -> None:
        if cfg.port <= 0:
            raise ValueError(f"Observation publisher port must be positive, got {cfg.port}")
        self.cfg = cfg
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_published_image_key: tuple[int | None, int] | None = None

    def publish(self, payload: dict[str, Any]) -> None:
        if self.cfg.image_store_key is not None:
            store = get_observation_image_store(self.cfg.image_store_key)
            image = None if store is None else store.latest()
            image_key = (
                None if image is None else (image.frame, len(image.data_base64))
            )
            if image is not None and image_key != self._last_published_image_key:
                payload = {
                    **payload,
                    "image": {
                        "mime_type": image.mime_type,
                        "data_base64": image.data_base64,
                        "frame": image.frame,
                        "width": image.width,
                        "height": image.height,
                    },
                }
                self._last_published_image_key = image_key
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._sock.sendto(data, (self.cfg.host, self.cfg.port))

    def close(self) -> None:
        self._sock.close()


def make_online_textop_observation(
    *,
    frame: int,
    started: bool,
    current_frame: int,
    latest_frame: int | None,
    lag_frames: int,
    buffer_frames: int,
    stale_steps: int,
    consecutive_stale_steps: int,
    robot_anchor_pos_w: Any,
    robot_anchor_quat_w: Any,
    fallen: bool = False,
    fall_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "mjlab_textop.online_observation.v1",
        "frame": int(frame),
        "started": bool(started),
        "current_frame": int(current_frame),
        "latest_frame": None if latest_frame is None else int(latest_frame),
        "lag_frames": int(lag_frames),
        "buffer_frames": int(buffer_frames),
        "stale_steps": int(stale_steps),
        "consecutive_stale_steps": int(consecutive_stale_steps),
        "fallen": bool(fallen),
        "fall_reason": fall_reason,
        "robot_anchor_pos_w": [
            float(item)
            for item in robot_anchor_pos_w.detach().cpu().reshape(-1).tolist()
        ],
        "robot_anchor_quat_w": [
            float(item)
            for item in robot_anchor_quat_w.detach().cpu().reshape(-1).tolist()
        ],
    }
