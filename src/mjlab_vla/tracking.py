from __future__ import annotations

from pathlib import Path

from mjlab.tasks.tracking.mdp import MotionCommandCfg

TASK_NAME = "Mjlab-Tracking-Flat-Unitree-G1"


def get_motion_command_cfg(commands) -> MotionCommandCfg:
    motion_cmd = commands["motion"]
    if not isinstance(motion_cmd, MotionCommandCfg):
        raise TypeError(
            "Expected env_cfg.commands['motion'] to be a MotionCommandCfg, "
            f"got {type(motion_cmd).__name__}"
        )
    return motion_cmd


def set_motion_file(env_cfg, motion_file: Path) -> None:
    motion_cmd = get_motion_command_cfg(env_cfg.commands)
    motion_cmd.motion_file = str(motion_file)
