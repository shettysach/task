from __future__ import annotations

from pathlib import Path

from mjlab_textop.core.motion import (
    load_mjlab_motion,
    reindex_mjlab_g1_joints_to_textop,
)
from mjlab_textop.core.online.source import (
    QueueTextOpOnlineSource,
    TextOpMotionBlock,
)
from mjlab_textop.core.schema import TEXTOP_ROOT_BODY_INDEX


def make_mjlab_npz_replay_source(
    path: str | Path,
    *,
    block_size: int = 8,
) -> QueueTextOpOnlineSource:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    motion = load_mjlab_motion(path)
    joint_pos_textop = reindex_mjlab_g1_joints_to_textop(motion.joint_pos)
    joint_vel_textop = reindex_mjlab_g1_joints_to_textop(motion.joint_vel)
    anchor_pos_w = motion.body_pos_w[:, TEXTOP_ROOT_BODY_INDEX]
    anchor_quat_w = motion.body_quat_w[:, TEXTOP_ROOT_BODY_INDEX]

    blocks = []
    for start in range(0, motion.num_frames, block_size):
        stop = min(start + block_size, motion.num_frames)
        blocks.append(
            TextOpMotionBlock(
                index=start,
                joint_pos=joint_pos_textop[start:stop],
                joint_vel=joint_vel_textop[start:stop],
                anchor_pos_w=anchor_pos_w[start:stop],
                anchor_quat_w=anchor_quat_w[start:stop],
            )
        )
    return QueueTextOpOnlineSource(blocks, fps=motion.fps)
