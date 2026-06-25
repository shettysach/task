from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import tyro

from mjlab_textop.core.normalize import normalize_textop_npz
from mjlab_textop.core.normalize_robotmdar_record import normalize_robotmdar_record_npz
from mjlab_textop.scripts.eval import EvalCommand, evaluate_textop_motion
from mjlab_textop.scripts.normalize import NormalizeCommand
from mjlab_textop.scripts.normalize_robotmdar_record import (
    NormalizeRobotMdarRecordCommand,
)
from mjlab_textop.scripts.play_live import PlayLiveCommand, play_live_textop_motion
from mjlab_textop.scripts.play_online import (
    PlayOnlineCommand,
    play_online_textop_motion,
)

TextOpCommand: TypeAlias = (
    NormalizeCommand
    | NormalizeRobotMdarRecordCommand
    | PlayOnlineCommand
    | PlayLiveCommand
    | EvalCommand
)

TextOpCommandType = tyro.extras.subcommand_type_from_defaults(
    {
        "normalize": NormalizeCommand(),
        "normalize-robotmdar-record": NormalizeRobotMdarRecordCommand(),
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
        case NormalizeCommand():
            input_file = verify_path(cfg.motion_file, "TextOp motion file")
            output_file = resolve_path(cfg.normalized_motion_file)
            normalize_textop_npz(input_file, output_file, device=cfg.device)
            return

        case NormalizeRobotMdarRecordCommand():
            input_file = verify_path(cfg.recorded_motion_file, "RobotMDAR raw record")
            output_file = resolve_path(cfg.normalized_motion_file)
            normalize_robotmdar_record_npz(
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
