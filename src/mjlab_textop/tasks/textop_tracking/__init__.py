from __future__ import annotations

from mjlab_textop.tasks.textop_tracking.env_cfg import (
    make_textop_g1_flat_tracking_env_cfg,
)
from mjlab_textop.tasks.textop_tracking.registration import (
    STATIC_TASK_SPECS,
    TEXTOP_TASK_NAME,
)

__all__ = [
    "STATIC_TASK_SPECS",
    "TEXTOP_TASK_NAME",
    "make_textop_g1_flat_tracking_env_cfg",
]
