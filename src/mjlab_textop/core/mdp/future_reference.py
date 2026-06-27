from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FutureReferenceCommand(Protocol):
    @property
    def future_joint_pos(self) -> torch.Tensor: ...

    @property
    def future_joint_vel(self) -> torch.Tensor: ...

    @property
    def future_anchor_pos_w(self) -> torch.Tensor: ...

    @property
    def future_anchor_quat_w(self) -> torch.Tensor: ...

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor: ...

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor: ...
