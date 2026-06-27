from __future__ import annotations

from typing import Any

import numpy as np

from mjlab_textop.core.motion import (
    reindex_mjlab_g1_joints_to_textop,
)
from mjlab_textop.core.online.source import TextOpMotionBlock
from mjlab_textop.core.schema import MJLAB_G1_JOINT_NAMES, TEXTOP_G1_JOINT_COUNT

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


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
