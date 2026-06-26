from __future__ import annotations

import json
import socket
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from mjlab_textop.core.motion import (
    normalize_quat,
    validate_frame_vector_array,
    validate_g1_joint_frames,
)
from mjlab_textop.core.online.source import TextOpMotionBlock


@dataclass(frozen=True)
class SocketTextOpSourceCfg:
    host: str = "127.0.0.1"
    port: int = 8765
    fps: float = 50.0
    max_queue_blocks: int = 32


@dataclass
class TextOpLiveDiagnostics:
    queue_depth: int = 0
    blocks_received: int = 0
    blocks_polled: int = 0
    blocks_dropped: int = 0
    bad_messages: int = 0
    last_error: str | None = None
    connected: bool = False


def textop_block_to_ndjson_message(
    block: TextOpMotionBlock,
    *,
    fps: float = 50.0,
) -> str:
    message = {
        "index": int(block.index),
        "fps": float(fps),
        "joint_pos": np.asarray(block.joint_pos, dtype=np.float32).tolist(),
        "joint_vel": np.asarray(block.joint_vel, dtype=np.float32).tolist(),
        "anchor_pos_w": np.asarray(block.anchor_pos_w, dtype=np.float32).tolist(),
        "anchor_quat_w": np.asarray(block.anchor_quat_w, dtype=np.float32).tolist(),
    }
    return json.dumps(message, separators=(",", ":")) + "\n"


def parse_textop_block_message(
    message: str | bytes | dict[str, Any],
    *,
    default_fps: float = 50.0,
) -> tuple[TextOpMotionBlock, float]:
    data = _load_message(message)
    missing = [
        key
        for key in (
            "index",
            "joint_pos",
            "joint_vel",
            "anchor_pos_w",
            "anchor_quat_w",
        )
        if key not in data
    ]
    if missing:
        raise ValueError(f"TextOp live block missing required fields: {missing}")

    index = int(data["index"])
    joint_pos = validate_g1_joint_frames("joint_pos", data["joint_pos"])
    joint_vel = validate_g1_joint_frames("joint_vel", data["joint_vel"])
    anchor_pos_w = validate_frame_vector_array("anchor_pos_w", data["anchor_pos_w"], 3)
    anchor_quat_w = normalize_quat(
        validate_frame_vector_array("anchor_quat_w", data["anchor_quat_w"], 4)
    )

    for name, value in (
        ("joint_vel", joint_vel),
        ("anchor_pos_w", anchor_pos_w),
        ("anchor_quat_w", anchor_quat_w),
    ):
        if value.shape[0] != joint_pos.shape[0]:
            raise ValueError(
                f"{name} frame count {value.shape[0]} differs from "
                f"joint_pos frame count {joint_pos.shape[0]}"
            )

    return (
        TextOpMotionBlock(
            index=index,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        ),
        float(data.get("fps", default_fps)),
    )


class SocketTextOpOnlineSource:
    def __init__(self, cfg: SocketTextOpSourceCfg | None = None) -> None:
        cfg = cfg or SocketTextOpSourceCfg()
        if cfg.max_queue_blocks <= 0:
            raise ValueError(
                f"max_queue_blocks must be positive, got {cfg.max_queue_blocks}"
            )
        self.cfg = cfg
        self.fps = cfg.fps
        self.diagnostics = TextOpLiveDiagnostics()
        self._queue: deque[TextOpMotionBlock] = deque()
        self._recorded_blocks: list[TextOpMotionBlock] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def poll(self) -> TextOpMotionBlock | None:
        with self._lock:
            if not self._queue:
                self._sync_queue_depth_locked()
                return None
            block = self._queue.popleft()
            self.diagnostics.blocks_polled += 1
            self._sync_queue_depth_locked()
            return block

    def append_message(self, message: str | bytes | dict[str, Any]) -> None:
        block, fps = parse_textop_block_message(message, default_fps=self.fps)
        self.fps = fps
        self._append_block(block)

    def recorded_blocks(self) -> list[TextOpMotionBlock]:
        with self._lock:
            return list(self._recorded_blocks)

    def _reader_loop(self) -> None:
        try:
            with socket.create_connection((self.cfg.host, self.cfg.port)) as sock:
                self._sock = sock
                self.diagnostics.connected = True
                with sock.makefile("rb") as reader:
                    for line in reader:
                        if self._stop.is_set():
                            return
                        self._handle_line(line)
        except OSError as exc:
            self.diagnostics.last_error = str(exc)
        finally:
            self.diagnostics.connected = False
            self._sock = None

    def _handle_line(self, line: bytes) -> None:
        try:
            self.append_message(line)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            with self._lock:
                self.diagnostics.bad_messages += 1
                self.diagnostics.last_error = str(exc)

    def _append_block(self, block: TextOpMotionBlock) -> None:
        with self._lock:
            if len(self._queue) >= self.cfg.max_queue_blocks:
                self._queue.popleft()
                self.diagnostics.blocks_dropped += 1
            self._queue.append(block)
            self._recorded_blocks.append(block)
            self.diagnostics.blocks_received += 1
            self._sync_queue_depth_locked()

    def _sync_queue_depth_locked(self) -> None:
        self.diagnostics.queue_depth = len(self._queue)


def _load_message(message: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    data = json.loads(message)
    if not isinstance(data, dict):
        raise ValueError("TextOp live block message must be a JSON object")
    return data
