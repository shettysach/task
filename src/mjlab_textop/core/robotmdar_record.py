from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_textop.core.motion import (
    normalize_quat,
    validate_frame_vector_array,
    validate_g1_joint_frames,
)
from mjlab_textop.core.online.source import TextOpMotionBlock

ROBOTMDAR_RAW_RECORD_REQUIRED_KEYS: tuple[str, ...] = (
    "fps",
    "joint_pos",
    "joint_vel",
    "anchor_pos_w",
    "anchor_quat_w",
)


@dataclass(frozen=True)
class RobotMdarRawRecord:
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray
    frame_index: np.ndarray
    prompt: str = ""
    guidance_scale: float | None = None
    num_blocks: int | None = None
    source: str = "robotmdar"

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


def save_robotmdar_raw_record(
    path: str | Path,
    blocks: Sequence[TextOpMotionBlock],
    *,
    fps: float,
    prompt: str,
    guidance_scale: float,
    source: str = "robotmdar",
) -> Path:
    """Save RobotMDAR online blocks as raw TextOp-order reference data."""

    if not blocks:
        raise ValueError("Cannot save an empty RobotMDAR raw record")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps value: {fps}")

    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for block in blocks:
        joint_pos = validate_g1_joint_frames("joint_pos", block.joint_pos)
        joint_vel = validate_g1_joint_frames("joint_vel", block.joint_vel)
        anchor_pos_w = validate_frame_vector_array(
            "anchor_pos_w", block.anchor_pos_w, 3
        )
        anchor_quat_w = normalize_quat(
            validate_frame_vector_array("anchor_quat_w", block.anchor_quat_w, 4)
        )
        _validate_matching_frame_counts(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )
        if block.index < 0:
            raise ValueError(
                f"TextOp block index must be non-negative, got {block.index}"
            )

        for offset in range(joint_pos.shape[0]):
            frame_index = block.index + offset
            if frame_index in frames:
                raise ValueError(f"Duplicate RobotMDAR raw frame index: {frame_index}")
            frames[frame_index] = (
                joint_pos[offset],
                joint_vel[offset],
                anchor_pos_w[offset],
                anchor_quat_w[offset],
            )

    ordered_indices = np.asarray(sorted(frames), dtype=np.int64)
    ordered = [frames[index] for index in ordered_indices]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([fps], dtype=np.float32),
        joint_pos=np.stack([frame[0] for frame in ordered], axis=0).astype(np.float32),
        joint_vel=np.stack([frame[1] for frame in ordered], axis=0).astype(np.float32),
        anchor_pos_w=np.stack([frame[2] for frame in ordered], axis=0).astype(
            np.float32
        ),
        anchor_quat_w=np.stack([frame[3] for frame in ordered], axis=0).astype(
            np.float32
        ),
        frame_index=ordered_indices,
        prompt=np.array(prompt),
        guidance_scale=np.array([guidance_scale], dtype=np.float32),
        num_blocks=np.array([len(blocks)], dtype=np.int64),
        source=np.array(source),
    )
    return output_path


def load_robotmdar_raw_record(path: str | Path) -> RobotMdarRawRecord:
    data = np.load(Path(path))
    missing = [key for key in ROBOTMDAR_RAW_RECORD_REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"RobotMDAR raw record is missing keys: {missing}")

    fps = _read_scalar_float(data, "fps")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps value: {fps}")

    joint_pos = validate_g1_joint_frames("joint_pos", data["joint_pos"])
    joint_vel = validate_g1_joint_frames("joint_vel", data["joint_vel"])
    anchor_pos_w = validate_frame_vector_array("anchor_pos_w", data["anchor_pos_w"], 3)
    anchor_quat_w = normalize_quat(
        validate_frame_vector_array("anchor_quat_w", data["anchor_quat_w"], 4)
    )
    _validate_matching_frame_counts(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        anchor_pos_w=anchor_pos_w,
        anchor_quat_w=anchor_quat_w,
    )

    frame_index = (
        np.asarray(data["frame_index"], dtype=np.int64)
        if "frame_index" in data
        else np.arange(joint_pos.shape[0], dtype=np.int64)
    )
    if frame_index.shape != (joint_pos.shape[0],):
        raise ValueError(
            f"frame_index must be shaped [{joint_pos.shape[0]}], got "
            f"{frame_index.shape}"
        )

    return RobotMdarRawRecord(
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        anchor_pos_w=anchor_pos_w,
        anchor_quat_w=anchor_quat_w,
        frame_index=frame_index,
        prompt=_read_optional_string(data, "prompt"),
        guidance_scale=(
            _read_scalar_float(data, "guidance_scale")
            if "guidance_scale" in data
            else None
        ),
        num_blocks=(
            int(np.asarray(data["num_blocks"]).reshape(-1)[0])
            if "num_blocks" in data
            else None
        ),
        source=_read_optional_string(data, "source") or "robotmdar",
    )


def _validate_matching_frame_counts(**arrays: np.ndarray) -> None:
    counts = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"RobotMDAR raw arrays have inconsistent frame counts: {counts}")


def _read_scalar_float(data: Any, key: str) -> float:
    value = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if value.size == 0:
        raise ValueError(f"Invalid {key} value: {data[key]}")
    return float(value[0])


def _read_optional_string(data: Any, key: str) -> str:
    if key not in data:
        return ""
    return str(np.asarray(data[key]).item())
