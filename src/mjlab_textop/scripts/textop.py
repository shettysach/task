from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import tyro

from mjlab_textop.core.normalize_robotmdar_npz import normalize_robotmdar_npz
from mjlab_textop.core.normalize_tracker_npz import normalize_tracker_npz
from mjlab_textop.scripts.eval import EvalCommand, evaluate_textop_motion
from mjlab_textop.scripts.normalize_robotmdar_npz import (
    NormalizeRobotMdarNpzCommand,
)
from mjlab_textop.scripts.normalize_tracker_npz import NormalizeTrackerNpzCommand
from mjlab_textop.scripts.play_live import PlayLiveCommand, play_live_textop_motion
from mjlab_textop.scripts.play_online import (
    PlayOnlineCommand,
    play_online_textop_motion,
)

TextOpCommand: TypeAlias = (
    NormalizeTrackerNpzCommand
    | NormalizeRobotMdarNpzCommand
    | PlayOnlineCommand
    | PlayLiveCommand
    | EvalCommand
)

TextOpCommandType = tyro.extras.subcommand_type_from_defaults(
    {
        "normalize-tracker-npz": NormalizeTrackerNpzCommand(),
        "normalize-robotmdar-npz": NormalizeRobotMdarNpzCommand(),
        "play-online": PlayOnlineCommand(),
        "play-live": PlayLiveCommand(),
        "eval": EvalCommand(),
    },
)


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def verify_resolved(resolved: Path, label: str) -> Path:
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def verify_path(path: str, label: str) -> Path:
    return verify_resolved(resolve_path(path), label)


def run_textop_motion(cfg: TextOpCommand) -> None:
    match cfg:
        case NormalizeTrackerNpzCommand():
            input_file = verify_path(cfg.motion_file, "TextOp motion file")
            output_file = resolve_path(cfg.normalized_motion_file)
            normalize_tracker_npz(input_file, output_file, device=cfg.device)
            return

        case NormalizeRobotMdarNpzCommand():
            input_file = verify_path(cfg.recorded_motion_file, "RobotMDAR raw record")
            output_file = resolve_path(cfg.normalized_motion_file)
            normalize_robotmdar_npz(
                input_file,
                output_file,
                device=cfg.device,
                max_frames=cfg.max_frames,
            )
            return

        case PlayOnlineCommand():
            motion_file = verify_path(
                cfg.motion_file,
                "Normalized motion file",
            )
            checkpoint_file = verify_path(
                cfg.checkpoint_file,
                "Checkpoint file",
            )
            play_online_textop_motion(
                cfg,
                motion_file=motion_file,
                checkpoint_file=checkpoint_file,
            )
            return

        case PlayLiveCommand():
            checkpoint_file = verify_path(
                cfg.checkpoint_file,
                "Checkpoint file",
            )
            play_live_textop_motion(
                cfg,
                checkpoint_file=checkpoint_file,
            )
            return

        case EvalCommand():
            motion_file = verify_path(
                cfg.motion_file,
                "Normalized motion file",
            )
            checkpoint_file = verify_path(
                cfg.checkpoint_file,
                "Checkpoint file",
            )
            evaluate_textop_motion(
                cfg,
                motion_file=motion_file,
                checkpoint_file=checkpoint_file,
            )
            return


def main() -> None:
    run_textop_motion(tyro.cli(TextOpCommandType))


if __name__ == "__main__":
    main()
