from __future__ import annotations

import numpy as np
import pytest

from mjlab_textop.core.contract import MJLAB_G1_JOINT_NAMES
from mjlab_textop.core.motion import (
    load_mjlab_motion,
    reindex_mjlab_g1_joints_to_textop,
    reindex_textop_g1_joints_to_mjlab,
)
from mjlab_textop.core.online.source import TextOpMotionBlock
from mjlab_textop.core.robotmdar import (
    ROBOTMDAR_G1_DOF_INDEX,
    ROBOTMDAR_G1_DOF_LINK_NAMES,
    ROBOTMDAR_G1_DOF_NAMES,
    expand_robotmdar_dof_to_mjlab_g1,
    robotmdar_motion_dict_to_block,
    save_textop_motion_blocks_as_mjlab_npz,
    slice_motion_dict_tail,
)


def test_robotmdar_dof_indices_are_derived_from_joint_names() -> None:
    assert ROBOTMDAR_G1_DOF_LINK_NAMES == (
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    )
    assert ROBOTMDAR_G1_DOF_INDEX == tuple(
        MJLAB_G1_JOINT_NAMES.index(name) for name in ROBOTMDAR_G1_DOF_NAMES
    )
    assert tuple(
        name
        for name in MJLAB_G1_JOINT_NAMES
        if name not in ROBOTMDAR_G1_DOF_NAMES
    ) == (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_expand_robotmdar_dof_to_mjlab_g1_places_known_dofs() -> None:
    robotmdar_dof = np.arange(2 * 23, dtype=np.float32).reshape(2, 23)

    mjlab_dof = expand_robotmdar_dof_to_mjlab_g1(robotmdar_dof)

    assert mjlab_dof.shape == (2, 29)
    np.testing.assert_allclose(mjlab_dof[:, ROBOTMDAR_G1_DOF_INDEX], robotmdar_dof)
    missing_joint_indices = [
        MJLAB_G1_JOINT_NAMES.index(name)
        for name in MJLAB_G1_JOINT_NAMES
        if name not in ROBOTMDAR_G1_DOF_NAMES
    ]
    np.testing.assert_allclose(mjlab_dof[:, missing_joint_indices], 0.0)


def test_expand_robotmdar_dof_to_mjlab_g1_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"Expected \[T, 23\] RobotMDAR DoF array"):
        expand_robotmdar_dof_to_mjlab_g1(np.zeros((2, 22), dtype=np.float32))


def test_robotmdar_motion_dict_to_block_converts_to_textop_block() -> None:
    dof_pos = np.arange(3 * 23, dtype=np.float32).reshape(1, 3, 23)
    dof_vel = dof_pos + 1000.0
    root_rot_xyzw = np.tile(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (1, 3, 1),
    )
    root_trans_offset = np.arange(9, dtype=np.float32).reshape(1, 3, 3)

    block = robotmdar_motion_dict_to_block(
        {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_rot": root_rot_xyzw,
            "root_trans_offset": root_trans_offset,
        },
        index=11,
    )

    expected_mjlab_pos = expand_robotmdar_dof_to_mjlab_g1(dof_pos[0])
    expected_mjlab_vel = expand_robotmdar_dof_to_mjlab_g1(dof_vel[0])
    assert block.index == 11
    np.testing.assert_allclose(
        block.joint_pos, reindex_mjlab_g1_joints_to_textop(expected_mjlab_pos)
    )
    np.testing.assert_allclose(
        block.joint_vel, reindex_mjlab_g1_joints_to_textop(expected_mjlab_vel)
    )
    np.testing.assert_allclose(block.anchor_pos_w, root_trans_offset[0])
    np.testing.assert_allclose(
        block.anchor_quat_w,
        np.tile(np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float32), (3, 1)),
    )


def test_slice_motion_dict_tail_slices_batched_time_arrays() -> None:
    batched = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)
    scalar = object()

    result = slice_motion_dict_tail({"batched": batched, "scalar": scalar}, 2)

    np.testing.assert_allclose(result["batched"], batched[:, -2:])
    assert result["scalar"] is scalar


def test_save_textop_motion_blocks_as_mjlab_npz_records_replay_motion(tmp_path) -> None:
    textop_joint_pos = np.arange(4 * 29, dtype=np.float32).reshape(4, 29)
    textop_joint_vel = textop_joint_pos + 1000.0
    anchor_pos_w = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    anchor_quat_w = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (4, 1),
    )
    blocks = [
        TextOpMotionBlock(
            index=2,
            joint_pos=textop_joint_pos[2:4],
            joint_vel=textop_joint_vel[2:4],
            anchor_pos_w=anchor_pos_w[2:4],
            anchor_quat_w=anchor_quat_w[2:4],
        ),
        TextOpMotionBlock(
            index=0,
            joint_pos=textop_joint_pos[0:2],
            joint_vel=textop_joint_vel[0:2],
            anchor_pos_w=anchor_pos_w[0:2],
            anchor_quat_w=anchor_quat_w[0:2],
        ),
    ]
    output_file = tmp_path / "robotmdar_recorded.npz"

    save_textop_motion_blocks_as_mjlab_npz(output_file, blocks, fps=50.0)

    motion = load_mjlab_motion(output_file)
    assert motion.fps == 50.0
    assert motion.joint_pos.shape == (4, 29)
    assert motion.joint_vel.shape == (4, 29)
    assert motion.body_pos_w.shape == (4, 1, 3)
    assert motion.body_quat_w.shape == (4, 1, 4)
    np.testing.assert_allclose(
        motion.joint_pos, reindex_textop_g1_joints_to_mjlab(textop_joint_pos)
    )
    np.testing.assert_allclose(
        motion.joint_vel, reindex_textop_g1_joints_to_mjlab(textop_joint_vel)
    )
    np.testing.assert_allclose(motion.body_pos_w[:, 0, :], anchor_pos_w)
    np.testing.assert_allclose(motion.body_quat_w[:, 0, :], anchor_quat_w)
