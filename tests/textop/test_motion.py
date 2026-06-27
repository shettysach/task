from __future__ import annotations

import numpy as np
import torch
from builders import write_mjlab_motion_npz
from mjlab.tasks.tracking.mdp.commands import MotionLoader

from mjlab_textop.core.motion import (
    load_mjlab_motion,
    reindex_textop_g1_joints_to_mjlab,
)
from mjlab_textop.core.schema import (
    MJLAB_G1_JOINT_NAMES,
    TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
)


def test_textop_to_mjlab_joint_index_matches_audited_textop_deploy_order():
    assert TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX == (
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    )
    assert len(MJLAB_G1_JOINT_NAMES) == 29


def test_reindex_textop_g1_joints_to_mjlab():
    textop_values = np.arange(29, dtype=np.float32)
    mjlab_values = reindex_textop_g1_joints_to_mjlab(textop_values)

    np.testing.assert_allclose(
        mjlab_values, textop_values[list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    )


def test_reindex_textop_g1_joints_to_mjlab_rejects_wrong_joint_count():
    textop_values = np.zeros((3, 28), dtype=np.float32)

    try:
        reindex_textop_g1_joints_to_mjlab(textop_values)
    except ValueError as exc:
        assert "Expected last joint dimension" in str(exc)
    else:
        raise AssertionError("Expected wrong joint count to be rejected")


def test_mjlab_motion_loader_accepts_normalized_npz(tmp_path):
    motion_file = tmp_path / "motion.npz"
    np.savez(
        motion_file,
        fps=np.array([50.0], dtype=np.float32),
        joint_pos=np.zeros((2, 29), dtype=np.float32),
        joint_vel=np.zeros((2, 29), dtype=np.float32),
        body_pos_w=np.zeros((2, 30, 3), dtype=np.float32),
        body_quat_w=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 30, 1)
        ),
        body_lin_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
    )

    loader = MotionLoader(
        str(motion_file),
        body_indexes=torch.tensor([0, 2, 4], dtype=torch.long),
        device="cpu",
    )

    assert loader.joint_pos.shape == (2, 29)
    assert loader.body_pos_w.shape == (2, 3, 3)


def test_load_mjlab_motion_accepts_normalized_npz(tmp_path):
    motion_file = tmp_path / "motion.npz"
    joint_pos, joint_vel, body_pos_w, body_quat_w = write_mjlab_motion_npz(
        motion_file,
        frames=2,
        bodies=30,
    )

    motion = load_mjlab_motion(motion_file)

    np.testing.assert_allclose(motion.joint_pos, joint_pos)
    np.testing.assert_allclose(motion.joint_vel, joint_vel)
    np.testing.assert_allclose(motion.body_pos_w, body_pos_w)
    np.testing.assert_allclose(motion.body_quat_w, body_quat_w)
    assert motion.num_frames == 2
