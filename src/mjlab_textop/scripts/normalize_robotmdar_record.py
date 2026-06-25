from __future__ import annotations

from dataclasses import dataclass, field

import tyro


@dataclass(kw_only=True)
class NormalizeRobotMdarRecordCommand:
    recorded_motion_file: str = field(default=tyro.MISSING)
    normalized_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None
