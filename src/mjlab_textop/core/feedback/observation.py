from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import imageio.v3 as iio


class TextOpObservationPublisher(Protocol):
    def publish(self, payload: dict[str, Any]) -> None:
        """Publish one MJLab observation payload."""


@dataclass(frozen=True)
class UdpObservationPublisherCfg:
    host: str = "127.0.0.1"
    port: int = 8766


@dataclass(frozen=True, kw_only=True)
class OnlineTextOpObservationCfg:
    publisher: TextOpObservationPublisher | None = None
    publish_interval: int = 1
    image_path: str | None = None
    image_publish_interval: int = 5

    def __post_init__(self) -> None:
        if self.publish_interval <= 0:
            raise ValueError(
                "publish_interval must be positive, "
                f"got {self.publish_interval}"
            )
        if self.image_publish_interval <= 0:
            raise ValueError(
                "image_publish_interval must be positive, "
                f"got {self.image_publish_interval}"
            )


class UdpObservationPublisher:
    def __init__(self, cfg: UdpObservationPublisherCfg) -> None:
        if cfg.port <= 0:
            raise ValueError(f"Observation publisher port must be positive, got {cfg.port}")
        self.cfg = cfg
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, payload: dict[str, Any]) -> None:
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
    image_path: str | None = None,
    image_frame: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "mjlab_textop.online_observation.v1",
        "frame": int(frame),
        "started": bool(started),
        "current_frame": int(current_frame),
        "latest_frame": None if latest_frame is None else int(latest_frame),
        "lag_frames": int(lag_frames),
        "buffer_frames": int(buffer_frames),
        "stale_steps": int(stale_steps),
        "consecutive_stale_steps": int(consecutive_stale_steps),
        "robot_anchor_pos_w": [
            float(item)
            for item in robot_anchor_pos_w.detach().cpu().reshape(-1).tolist()
        ],
        "robot_anchor_quat_w": [
            float(item)
            for item in robot_anchor_quat_w.detach().cpu().reshape(-1).tolist()
        ],
    }
    if image_path is not None:
        payload["image_path"] = image_path
        payload["image_frame"] = image_frame
    return payload


def write_render_image(path: str, image: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        dir=target.parent,
        delete=False,
    ) as tmp:
        iio.imwrite(tmp.name, image)
    os.replace(tmp.name, target)
