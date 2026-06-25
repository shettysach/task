from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_textop.core.contract import MJLAB_G1_JOINT_NAMES, TEXTOP_G1_JOINT_COUNT
from mjlab_textop.core.motion import (
    reindex_mjlab_g1_joints_to_textop,
    reindex_textop_g1_joints_to_mjlab,
    validate_frame_vector_array,
    validate_g1_joint_frames,
)
from mjlab_textop.core.online.source import TextOpMotionBlock

# RobotMDAR predicts 23 G1 DoFs.
# MJLab G1 has 29 joints; RobotMDAR does not output wrist joints.
ROBOTMDAR_G1_DOF_NAMES: tuple[str, ...] = (
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

ROBOTMDAR_G1_DOF_LINK_NAMES: tuple[str, ...] = tuple(
    "torso_link" if name == "waist_pitch_joint" else name.replace("_joint", "_link")
    for name in ROBOTMDAR_G1_DOF_NAMES
)

ROBOTMDAR_DOF_COUNT = len(ROBOTMDAR_G1_DOF_NAMES)

ROBOTMDAR_G1_DOF_INDEX: tuple[int, ...] = tuple(
    MJLAB_G1_JOINT_NAMES.index(name) for name in ROBOTMDAR_G1_DOF_NAMES
)


def expand_robotmdar_dof_to_mjlab_g1(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ROBOTMDAR_DOF_COUNT:
        raise ValueError(
            f"Expected [T, {ROBOTMDAR_DOF_COUNT}] RobotMDAR DoF array, "
            f"got {value.shape}"
        )

    out = np.zeros((value.shape[0], TEXTOP_G1_JOINT_COUNT), dtype=np.float32)
    out[:, ROBOTMDAR_G1_DOF_INDEX] = value
    return out


def robotmdar_motion_dict_to_block(
    motion_dict: dict[str, Any],
    *,
    index: int,
) -> TextOpMotionBlock:
    joint_pos_mjlab = expand_robotmdar_dof_to_mjlab_g1(
        _to_numpy(motion_dict["dof_pos"][0])
    )
    joint_vel_mjlab = expand_robotmdar_dof_to_mjlab_g1(
        _to_numpy(motion_dict["dof_vel"][0])
    )
    root_rot_xyzw = _to_numpy(motion_dict["root_rot"][0])

    return TextOpMotionBlock(
        index=index,
        joint_pos=reindex_mjlab_g1_joints_to_textop(joint_pos_mjlab),
        joint_vel=reindex_mjlab_g1_joints_to_textop(joint_vel_mjlab),
        anchor_pos_w=_to_numpy(motion_dict["root_trans_offset"][0]),
        anchor_quat_w=root_rot_xyzw[:, [3, 0, 1, 2]],
    )


def slice_motion_dict_tail(
    motion_dict: dict[str, Any],
    frames: int,
) -> dict[str, Any]:
    result = {}
    for key, value in motion_dict.items():
        if hasattr(value, "shape") and len(value.shape) >= 2:
            result[key] = value[:, -frames:]
        else:
            result[key] = value
    return result


def save_textop_motion_blocks_as_mjlab_npz(
    path: str | Path,
    blocks: Sequence[TextOpMotionBlock],
    *,
    fps: float,
) -> None:
    """Save TextOp-order online blocks as a normalized MJLab replay NPZ."""

    if not blocks:
        raise ValueError("Cannot record an empty RobotMDAR motion stream")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps value: {fps}")

    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for block in blocks:
        joint_pos = validate_g1_joint_frames("joint_pos", block.joint_pos)
        joint_vel = validate_g1_joint_frames("joint_vel", block.joint_vel)
        anchor_pos_w = validate_frame_vector_array(
            "anchor_pos_w", block.anchor_pos_w, 3
        )
        anchor_quat_w = validate_frame_vector_array(
            "anchor_quat_w", block.anchor_quat_w, 4
        )

        if block.index < 0:
            raise ValueError(
                f"TextOp block index must be non-negative, got {block.index}"
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

        joint_pos_mjlab = reindex_textop_g1_joints_to_mjlab(joint_pos)
        joint_vel_mjlab = reindex_textop_g1_joints_to_mjlab(joint_vel)
        for offset in range(joint_pos.shape[0]):
            frames[block.index + offset] = (
                joint_pos_mjlab[offset],
                joint_vel_mjlab[offset],
                anchor_pos_w[offset],
                anchor_quat_w[offset],
            )

    ordered = [frames[index] for index in sorted(frames)]
    joint_pos = np.stack([frame[0] for frame in ordered], axis=0).astype(np.float32)
    joint_vel = np.stack([frame[1] for frame in ordered], axis=0).astype(np.float32)
    body_pos_w = np.stack([frame[2] for frame in ordered], axis=0).astype(np.float32)[
        :, None, :
    ]
    body_quat_w = np.stack([frame[3] for frame in ordered], axis=0).astype(np.float32)[
        :, None, :
    ]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([fps], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
