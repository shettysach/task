from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeedbackObservation:
    frame: int
    started: bool
    current_frame: int
    latest_frame: int | None
    lag_frames: int
    buffer_frames: int
    stale_steps: int
    consecutive_stale_steps: int
    fallen: bool
    fall_reason: str | None
    robot_anchor_pos_w: tuple[float, float, float]
    robot_anchor_quat_w: tuple[float, float, float, float]


class UdpFeedbackReceiver:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        recv_buffer_bytes: int = 65535,
    ) -> None:
        if port <= 0:
            raise ValueError(f"Feedback port must be positive, got {port}")
        if recv_buffer_bytes <= 0:
            raise ValueError(
                f"recv_buffer_bytes must be positive, got {recv_buffer_bytes}"
            )
        self.host = host
        self.port = port
        self.recv_buffer_bytes = recv_buffer_bytes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._latest: FeedbackObservation | None = None
        self._latest_monotonic: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def latest(self) -> FeedbackObservation | None:
        with self._lock:
            return self._latest

    def latest_age_seconds(self) -> float | None:
        with self._lock:
            if self._latest_monotonic is None:
                return None
            return time.monotonic() - self._latest_monotonic

    def _recv_loop(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                self._sock = sock
                sock.bind((self.host, self.port))
                sock.settimeout(0.1)
                while not self._stop.is_set():
                    try:
                        data, _ = sock.recvfrom(self.recv_buffer_bytes)
                    except TimeoutError:
                        continue
                    except OSError as exc:
                        if not self._stop.is_set():
                            self.last_error = str(exc)
                        return
                    self._handle_packet(data)
        finally:
            self._sock = None

    def _handle_packet(self, data: bytes) -> None:
        try:
            observation = parse_feedback_observation(data)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            return
        with self._lock:
            self._latest = observation
            self._latest_monotonic = time.monotonic()


def parse_feedback_observation(
    message: bytes | str | dict[str, Any],
) -> FeedbackObservation:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if isinstance(message, str):
        data = json.loads(message)
    else:
        data = message
    if not isinstance(data, dict):
        raise ValueError("Feedback observation must be a JSON object")

    return FeedbackObservation(
        frame=int(data["frame"]),
        started=bool(data["started"]),
        current_frame=int(data["current_frame"]),
        latest_frame=(
            None if data.get("latest_frame") is None else int(data["latest_frame"])
        ),
        lag_frames=int(data["lag_frames"]),
        buffer_frames=int(data["buffer_frames"]),
        stale_steps=int(data["stale_steps"]),
        consecutive_stale_steps=int(data["consecutive_stale_steps"]),
        fallen=bool(data.get("fallen", False)),
        fall_reason=(
            None if data.get("fall_reason") is None else str(data["fall_reason"])
        ),
        robot_anchor_pos_w=_fixed_float_tuple(data["robot_anchor_pos_w"], 3),  # ty:ignore[invalid-argument-type]
        robot_anchor_quat_w=_fixed_float_tuple(data["robot_anchor_quat_w"], 4),  # ty:ignore[invalid-argument-type]
    )


def _fixed_float_tuple(value: Any, width: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"Expected sequence with {width} values, got {value!r}")
    return tuple(float(item) for item in value)
