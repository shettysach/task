from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class NormalizedMotionConfig:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
