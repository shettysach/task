from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mjlab_textop.core.motion import (
    normalize_quat,
    validate_frame_vector_array,
    validate_g1_joint_frames,
)


@dataclass(frozen=True)
class TextOpMotionBlock:
    """Block of online TextOp reference motion frames.

    Joint arrays are in TextOp/IsaacLab order. The rolling buffer converts them
    to MJLab order when appending.
    """

    index: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


def validate_textop_motion_block(block: TextOpMotionBlock) -> TextOpMotionBlock:
    joint_pos = validate_g1_joint_frames("joint_pos", block.joint_pos)
    joint_vel = validate_g1_joint_frames("joint_vel", block.joint_vel)
    anchor_pos_w = validate_frame_vector_array("anchor_pos_w", block.anchor_pos_w, 3)
    anchor_quat_w = normalize_quat(
        validate_frame_vector_array("anchor_quat_w", block.anchor_quat_w, 4)
    )

    if block.index < 0:
        raise ValueError(f"TextOp block index must be non-negative, got {block.index}")
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

    return TextOpMotionBlock(
        index=block.index,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        anchor_pos_w=anchor_pos_w,
        anchor_quat_w=anchor_quat_w,
    )


class TextOpOnlineSource(Protocol):
    def poll(self) -> TextOpMotionBlock | None:
        """Return the next available block, or None when no block is ready."""


@runtime_checkable
class ResettableTextOpOnlineSource(TextOpOnlineSource, Protocol):
    def reset(self) -> None:
        """Reset a finite source to its initial frame."""


class QueueTextOpOnlineSource:
    def __init__(
        self,
        blocks: list[TextOpMotionBlock] | None = None,
        *,
        fps: float | None = None,
    ) -> None:
        self._initial_blocks = tuple(blocks or [])
        self._blocks: deque[TextOpMotionBlock] = deque(self._initial_blocks)
        self.fps = fps

    def append(self, block: TextOpMotionBlock) -> None:
        self._blocks.append(block)

    def poll(self) -> TextOpMotionBlock | None:
        if not self._blocks:
            return None
        return self._blocks.popleft()

    def reset(self) -> None:
        self._blocks = deque(self._initial_blocks)
