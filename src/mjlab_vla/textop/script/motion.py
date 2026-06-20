from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab_vla.textop.contract import (
    MJLAB_G1_JOINT_NAMES,
    TEXTOP_G1_JOINT_COUNT,
    TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
    TEXTOP_OPTIONAL_INPUT_KEYS,
    TEXTOP_REQUIRED_INPUT_KEYS,
    TEXTOP_ROOT_BODY_INDEX,
)

__all__ = (
    "MJLAB_G1_JOINT_NAMES",
    "TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX",
    "TextOpMotion",
    "load_textop_motion",
    "reindex_textop_g1_joints_to_mjlab",
)


@dataclass(frozen=True)
class TextOpMotion:
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


def reindex_textop_g1_joints_to_mjlab(values: np.ndarray) -> np.ndarray:
    """Convert G1 joint arrays from TextOp IsaacLab order to MJLab/MuJoCo order."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            "Expected last joint dimension to be "
            f"{TEXTOP_G1_JOINT_COUNT}, got {values.shape[-1]}"
        )
    return values[..., TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX]


def load_textop_motion(path: str | Path, fps: float | None = None) -> TextOpMotion:
    """Load a canonical TextOp tracker NPZ and normalize joint order for MJLab."""

    data = np.load(Path(path))
    resolved_fps = _resolve_fps(data, fps)
    _require_keys(data, TEXTOP_REQUIRED_INPUT_KEYS)

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    _validate_joint_array("joint_pos", joint_pos)
    _validate_joint_array("joint_vel", joint_vel)

    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = _normalize_quat(np.asarray(data["body_quat_w"], dtype=np.float32))
    _validate_body_arrays(body_pos_w, body_quat_w)
    _validate_optional_body_velocity_arrays(data, body_pos_w)

    root_pos_w = body_pos_w[:, TEXTOP_ROOT_BODY_INDEX].astype(np.float32)
    root_quat_w = body_quat_w[:, TEXTOP_ROOT_BODY_INDEX].astype(np.float32)
    root_lin_vel_w = _read_root_body_velocity(
        data, "body_lin_vel_w", body_pos_w, resolved_fps
    )
    root_ang_vel_w = _read_root_body_velocity(
        data, "body_ang_vel_w", body_pos_w, resolved_fps
    )

    motion = TextOpMotion(
        fps=resolved_fps,
        joint_pos=reindex_textop_g1_joints_to_mjlab(joint_pos),
        joint_vel=reindex_textop_g1_joints_to_mjlab(joint_vel),
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
    )
    _validate_frame_count(
        {
            "joint_pos": motion.joint_pos,
            "joint_vel": motion.joint_vel,
            "root_pos_w": motion.root_pos_w,
            "root_quat_w": motion.root_quat_w,
            "root_lin_vel_w": motion.root_lin_vel_w,
            "root_ang_vel_w": motion.root_ang_vel_w,
        }
    )
    return motion


def _resolve_fps(data: np.lib.npyio.NpzFile, fps: float | None) -> float:
    if fps is not None:
        return _validate_fps_value(fps)
    if "fps" not in data:
        raise ValueError("TextOp NPZ must contain `fps`, or pass an explicit fps value")
    fps_array = np.asarray(data["fps"], dtype=np.float32).reshape(-1)
    if fps_array.size == 0:
        raise ValueError(f"Invalid fps value: {data['fps']}")
    return _validate_fps_value(float(fps_array[0]))


def _validate_fps_value(fps: float) -> float:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps value: {fps}")
    return float(fps)


def _require_keys(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"TextOp NPZ is missing required keys: {missing}")


def _validate_joint_array(name: str, value: np.ndarray) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be shaped [T, J], got {value.shape}")
    if value.shape[-1] != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            f"{name} must have {TEXTOP_G1_JOINT_COUNT} joints, got {value.shape[-1]}"
        )


def _validate_body_arrays(body_pos_w: np.ndarray, body_quat_w: np.ndarray) -> None:
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must be shaped [T, B, 3], got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(
            f"body_quat_w must be shaped [T, B, 4], got {body_quat_w.shape}"
        )
    if body_pos_w.shape[:2] != body_quat_w.shape[:2]:
        raise ValueError(
            f"body_pos_w/body_quat_w frame-body shapes differ: "
            f"{body_pos_w.shape[:2]} vs {body_quat_w.shape[:2]}"
        )
    if body_pos_w.shape[1] <= TEXTOP_ROOT_BODY_INDEX:
        raise ValueError(
            f"body arrays must include root body index {TEXTOP_ROOT_BODY_INDEX}, "
            f"got {body_pos_w.shape[1]} bodies"
        )


def _validate_optional_body_velocity_arrays(
    data: np.lib.npyio.NpzFile, body_pos_w: np.ndarray
) -> None:
    for key in TEXTOP_OPTIONAL_INPUT_KEYS:
        if key == "fps":
            continue
        if key not in data:
            continue
        value = np.asarray(data[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{key} must be shaped [T, B, 3], got {value.shape}")
        if value.shape[:2] != body_pos_w.shape[:2]:
            raise ValueError(
                f"{key}/body_pos_w frame-body shapes differ: "
                f"{value.shape[:2]} vs {body_pos_w.shape[:2]}"
            )


def _read_root_body_velocity(
    data: np.lib.npyio.NpzFile,
    key: str,
    body_pos_w: np.ndarray,
    fps: float,
) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=np.float32)[:, TEXTOP_ROOT_BODY_INDEX]

    if key == "body_lin_vel_w":
        return _finite_difference_linear_velocity(
            body_pos_w[:, TEXTOP_ROOT_BODY_INDEX].astype(np.float32), fps
        )
    return np.zeros_like(body_pos_w[:, TEXTOP_ROOT_BODY_INDEX], dtype=np.float32)


def _finite_difference_linear_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[0] > 1:
        vel[:-1] = (pos[1:] - pos[:-1]) * fps
        vel[-1] = vel[-2]
    return vel


def _validate_frame_count(arrays: dict[str, np.ndarray]) -> None:
    counts = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Motion arrays have inconsistent frame counts: {counts}")


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Quaternion arrays contain zero-norm entries")
    return (quat / norm).astype(np.float32)
