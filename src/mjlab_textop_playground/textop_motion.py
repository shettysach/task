from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MJLAB_G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Audited against TextOp's deploy/replay scripts. TextOp stores G1 joints in
# IsaacLab policy order; MJLab/MuJoCo consumes them in XML joint order.
# fmt: off
TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX: tuple[int, ...] = (
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
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
    if values.shape[-1] != len(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX):
        raise ValueError(
            "Expected last joint dimension to be "
            f"{len(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)}, got {values.shape[-1]}"
        )
    return values[..., TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX]


def load_textop_motion(path: str | Path, fps: float | None = None) -> TextOpMotion:
    """Load a canonical TextOp tracker NPZ and normalize joint order for MJLab."""

    data = np.load(Path(path))
    resolved_fps = float(fps if fps is not None else _read_fps(data))
    _require_keys(
        data,
        (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ),
    )

    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = _normalize_quat(np.asarray(data["body_quat_w"], dtype=np.float32))
    _validate_body_arrays(body_pos_w, body_quat_w)

    motion = TextOpMotion(
        fps=resolved_fps,
        joint_pos=reindex_textop_g1_joints_to_mjlab(data["joint_pos"]),
        joint_vel=reindex_textop_g1_joints_to_mjlab(data["joint_vel"]),
        root_pos_w=body_pos_w[:, 0].astype(np.float32),
        root_quat_w=body_quat_w[:, 0].astype(np.float32),
        root_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, 0],
        root_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, 0],
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


def _read_fps(data: np.lib.npyio.NpzFile) -> float:
    if "fps" not in data:
        raise ValueError("TextOp NPZ must contain `fps`, or pass an explicit fps value")
    fps_array = np.asarray(data["fps"], dtype=np.float32).reshape(-1)
    if fps_array.size == 0 or fps_array[0] <= 0:
        raise ValueError(f"Invalid fps value: {data['fps']}")
    return float(fps_array[0])


def _require_keys(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"TextOp NPZ is missing required keys: {missing}")


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


def _validate_frame_count(arrays: dict[str, np.ndarray]) -> None:
    counts = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Motion arrays have inconsistent frame counts: {counts}")


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Quaternion arrays contain zero-norm entries")
    return (quat / norm).astype(np.float32)
