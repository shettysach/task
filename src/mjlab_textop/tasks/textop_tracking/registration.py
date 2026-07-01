from __future__ import annotations

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.task import StaticTaskSpec
from mjlab_textop.tasks.textop_tracking.env_cfg import (
    make_textop_g1_flat_tracking_env_cfg,
)
from mjlab_textop.tasks.textop_tracking.ppo_cfg import (
    unitree_g1_tracking_ppo_runner_cfg,
)

TEXTOP_TASK_NAME = "Mjlab-TextOp-Flat-Unitree-G1"

STATIC_TASK_SPECS = [
    StaticTaskSpec(
        task_id=TEXTOP_TASK_NAME,
        make_env_cfg=lambda: make_textop_g1_flat_tracking_env_cfg(play=False),
        make_play_env_cfg=lambda: make_textop_g1_flat_tracking_env_cfg(play=True),
        make_rl_cfg=unitree_g1_tracking_ppo_runner_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    ),
]
