from __future__ import annotations

from mjlab_textop.core.task import register_static_task_specs
from mjlab_textop.tasks.online_textop.registration import (
    STATIC_TASK_SPECS as ONLINE_TEXTOP_TASK_SPECS,
)
from mjlab_textop.tasks.textop_tracking.registration import (
    STATIC_TASK_SPECS as TEXTOP_TRACKING_TASK_SPECS,
)


def register_tasks() -> None:
    register_static_task_specs(
        [
            *TEXTOP_TRACKING_TASK_SPECS,
            *ONLINE_TEXTOP_TASK_SPECS,
        ]
    )


def ensure_textop_task_registered() -> None:
    register_tasks()
