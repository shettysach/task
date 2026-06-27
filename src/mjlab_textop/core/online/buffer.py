from __future__ import annotations

import torch

from mjlab_textop.core.motion import (
    reindex_textop_g1_joints_to_mjlab,
)
from mjlab_textop.core.online.source import (
    TextOpMotionBlock,
    validate_textop_motion_block,
)


class TextOpRollingMotionBuffer:
    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        max_frames: int | None = 512,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self.device = torch.device(device)
        self.max_frames = max_frames
        self._joint_pos: dict[int, torch.Tensor] = {}
        self._joint_vel: dict[int, torch.Tensor] = {}
        self._anchor_pos_w: dict[int, torch.Tensor] = {}
        self._anchor_quat_w: dict[int, torch.Tensor] = {}
        self._latest_index: int | None = None

    @property
    def latest_index(self) -> int | None:
        return self._latest_index

    @property
    def earliest_index(self) -> int | None:
        if not self._joint_pos:
            return None
        return min(self._joint_pos)

    @property
    def frame_count(self) -> int:
        return len(self._joint_pos)

    def clear(self) -> None:
        self._joint_pos.clear()
        self._joint_vel.clear()
        self._anchor_pos_w.clear()
        self._anchor_quat_w.clear()
        self._latest_index = None

    def append_block(self, block: TextOpMotionBlock) -> None:
        block = validate_textop_motion_block(block)

        joint_pos = reindex_textop_g1_joints_to_mjlab(block.joint_pos)
        joint_vel = reindex_textop_g1_joints_to_mjlab(block.joint_vel)

        for offset in range(joint_pos.shape[0]):
            frame = block.index + offset
            self._joint_pos[frame] = torch.as_tensor(
                joint_pos[offset], dtype=torch.float32, device=self.device
            )
            self._joint_vel[frame] = torch.as_tensor(
                joint_vel[offset], dtype=torch.float32, device=self.device
            )
            self._anchor_pos_w[frame] = torch.as_tensor(
                block.anchor_pos_w[offset], dtype=torch.float32, device=self.device
            )
            self._anchor_quat_w[frame] = torch.as_tensor(
                block.anchor_quat_w[offset], dtype=torch.float32, device=self.device
            )

        block_latest = block.index + joint_pos.shape[0] - 1
        self._latest_index = (
            block_latest
            if self._latest_index is None
            else max(self._latest_index, block_latest)
        )
        self._evict_old_frames()

    def can_start(self, frame: int, future_steps: int) -> bool:
        return all(
            (frame + offset) in self._joint_pos for offset in range(future_steps)
        )

    def earliest_start_frame(self, future_steps: int) -> int | None:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        for frame in sorted(self._joint_pos):
            if self.can_start(frame, future_steps):
                return frame
        return None

    def latest_start_frame(self, future_steps: int) -> int | None:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        for frame in sorted(self._joint_pos, reverse=True):
            if self.can_start(frame, future_steps):
                return frame
        return None

    def get_future(
        self,
        frame: int,
        future_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        if not self._joint_pos:
            raise RuntimeError("Online TextOp buffer has no frames")

        stale_steps = 0
        frames = []
        for offset in range(future_steps):
            requested = frame + offset
            resolved = self._resolve_frame(requested)
            if resolved != requested:
                stale_steps += 1
            frames.append(resolved)

        return (
            torch.stack([self._joint_pos[idx] for idx in frames], dim=0),
            torch.stack([self._joint_vel[idx] for idx in frames], dim=0),
            torch.stack([self._anchor_pos_w[idx] for idx in frames], dim=0),
            torch.stack([self._anchor_quat_w[idx] for idx in frames], dim=0),
            stale_steps,
        )

    def _resolve_frame(self, frame: int) -> int:
        if frame in self._joint_pos:
            return frame

        available = [idx for idx in self._joint_pos if idx <= frame]
        if available:
            return max(available)

        raise RuntimeError(f"No available online TextOp frame at or before {frame}")

    def _evict_old_frames(self) -> None:
        if self.max_frames is None or self._latest_index is None:
            return
        first_kept = self._latest_index - self.max_frames + 1
        for frame in list(self._joint_pos):
            if frame < first_kept:
                del self._joint_pos[frame]
                del self._joint_vel[frame]
                del self._anchor_pos_w[frame]
                del self._anchor_quat_w[frame]
