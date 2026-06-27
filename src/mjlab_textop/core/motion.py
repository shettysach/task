from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab_textop.core.schema import (
    TEXTOP_G1_JOINT_COUNT,
    TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
    TEXTOP_ROOT_BODY_INDEX,
)

MJLAB_TO_TEXTOP_G1_JOINT_INDEX: tuple[int, ...] = tuple(
    int(i) for i in np.argsort(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
)

MJLAB_REQUIRED_INPUT_KEYS: tuple[str, ...] = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
)


@dataclass(frozen=True)
class MjlabMotion:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    fps: float | None = None

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def root_pos_w(self) -> np.ndarray:
        return self.body_pos_w[:, TEXTOP_ROOT_BODY_INDEX]

    @property
    def root_quat_w(self) -> np.ndarray:
        return self.body_quat_w[:, TEXTOP_ROOT_BODY_INDEX]


def reindex_textop_g1_joints_to_mjlab(values: np.ndarray) -> np.ndarray:
    """Convert G1 joint arrays from TextOp IsaacLab order to MJLab/MuJoCo order."""

    values = validate_g1_joint_last_dim("values", values)
    return values[..., TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX]


def reindex_mjlab_g1_joints_to_textop(values: np.ndarray) -> np.ndarray:
    """Convert G1 joint arrays from MJLab/MuJoCo order to TextOp IsaacLab order."""

    values = validate_g1_joint_last_dim("values", values)
    return values[..., MJLAB_TO_TEXTOP_G1_JOINT_INDEX]


def load_mjlab_motion(path: str | Path) -> MjlabMotion:
    """Load a normalized MJLab tracking NPZ.

    Joint arrays are expected to already be in MJLab/MuJoCo order.
    """

    data = np.load(Path(path))
    _require_keys(data, MJLAB_REQUIRED_INPUT_KEYS)

    motion = MjlabMotion(
        fps=_resolve_optional_fps(data),
        joint_pos=validate_g1_joint_frames("joint_pos", data["joint_pos"]),
        joint_vel=validate_g1_joint_frames("joint_vel", data["joint_vel"]),
        body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float32),
        body_quat_w=normalize_quat(np.asarray(data["body_quat_w"], dtype=np.float32)),
    )
    _validate_body_arrays(motion.body_pos_w, motion.body_quat_w)
    _validate_frame_count(
        {
            "joint_pos": motion.joint_pos,
            "joint_vel": motion.joint_vel,
            "body_pos_w": motion.body_pos_w,
            "body_quat_w": motion.body_quat_w,
        }
    )
    return motion


def _resolve_optional_fps(data: np.lib.npyio.NpzFile) -> float | None:
    if "fps" not in data:
        return None
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
        raise ValueError(f"Motion NPZ is missing required keys: {missing}")


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
    if body_pos_w.shape[0] == 0:
        raise ValueError("body arrays must contain at least one frame")
    if not np.all(np.isfinite(body_pos_w)) or not np.all(np.isfinite(body_quat_w)):
        raise ValueError("body arrays contain non-finite values")


def validate_g1_joint_last_dim(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            "Expected last joint dimension to be "
            f"{TEXTOP_G1_JOINT_COUNT}, got {array.shape[-1]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_g1_joint_frames(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be shaped [T, J], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if array.shape[-1] != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            f"{name} must have {TEXTOP_G1_JOINT_COUNT} joints, got {array.shape[-1]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_body_vector_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be shaped [T, B, 3], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_frame_vector_array(
    name: str,
    value: np.ndarray,
    width: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be shaped [T, {width}], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_frame_count(arrays: dict[str, np.ndarray]) -> None:
    counts = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Motion arrays have inconsistent frame counts: {counts}")


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion arrays contain non-finite values")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Quaternion arrays contain zero-norm entries")
    return (quat / norm).astype(np.float32)
