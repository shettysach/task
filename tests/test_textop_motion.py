from __future__ import annotations

import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader

from mjlab_textop_playground.textop_motion import (
  MJLAB_G1_JOINT_NAMES,
  TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
  load_textop_motion,
  reindex_textop_g1_joints_to_mjlab,
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


def test_load_textop_motion_reindexes_canonical_textop_tracker_npz(tmp_path):
  frames = 3
  textop_joint_pos = np.stack(
    [
      np.arange(29, dtype=np.float32),
      np.arange(29, dtype=np.float32) + 1.0,
      np.arange(29, dtype=np.float32) + 2.0,
    ],
    axis=0,
  )
  textop_joint_vel = textop_joint_pos + 10.0
  body_pos_w = np.zeros((frames, 1, 3), dtype=np.float32)
  body_pos_w[:, 0, 0] = np.array([0.0, 0.1, 0.2], dtype=np.float32)
  body_quat_w = np.zeros((frames, 1, 4), dtype=np.float32)
  body_quat_w[:, 0, 0] = 1.0
  body_lin_vel_w = np.zeros((frames, 1, 3), dtype=np.float32)
  body_lin_vel_w[:, 0, 0] = 1.0
  body_ang_vel_w = np.zeros((frames, 1, 3), dtype=np.float32)

  input_file = tmp_path / "textop.npz"
  np.savez(
    input_file,
    fps=np.array([10.0], dtype=np.float32),
    joint_pos=textop_joint_pos,
    joint_vel=textop_joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )

  motion = load_textop_motion(input_file)

  assert motion.fps == 10.0
  np.testing.assert_allclose(
    motion.joint_pos, reindex_textop_g1_joints_to_mjlab(textop_joint_pos)
  )
  np.testing.assert_allclose(
    motion.joint_vel, reindex_textop_g1_joints_to_mjlab(textop_joint_vel)
  )
  np.testing.assert_allclose(motion.root_lin_vel_w, np.tile([1.0, 0.0, 0.0], (3, 1)))
  np.testing.assert_allclose(motion.root_ang_vel_w, np.zeros((3, 3)))


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
