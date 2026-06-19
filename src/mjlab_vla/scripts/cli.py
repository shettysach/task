from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import tyro

from mjlab_vla.scripts.eval_textop_motion import EvalCommand, evaluate_textop_motion
from mjlab_vla.scripts.normalize_textop_motion import (
    NormalizeCommand,
    normalize_textop_npz,
)
from mjlab_vla.scripts.play_textop_motion import PlayCommand, play_textop_motion
from mjlab_vla.scripts.train_textop_motion import TrainCommand, train_textop_motion

TextOpCommand: TypeAlias = NormalizeCommand | TrainCommand | PlayCommand | EvalCommand

TextOpCommandType = tyro.extras.subcommand_type_from_defaults(
    {
        "normalize": NormalizeCommand(),
        "train": TrainCommand(),
        "play": PlayCommand(),
        "eval": EvalCommand(),
    },
)


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def require_existing_file(path: str, label: str) -> Path:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def run_textop_motion(cfg: TextOpCommand) -> None:
    match cfg:
        case NormalizeCommand():
            # TODO: Simplify
            input_path = str(resolve_path(cfg.data_dir) / cfg.motion_rel)
            input_file = require_existing_file(
                input_path,
                "TextOp motion file",
            )
            output_file = resolve_path(cfg.normalized_motion_file)
            normalize_textop_npz(input_file, output_file, device=cfg.device)
            return

        case TrainCommand():
            motion_file = require_existing_file(
                cfg.normalized_motion_file,
                "Normalized motion file",
            )
            train_textop_motion(cfg, motion_file=motion_file)
            return

        case PlayCommand():
            motion_file = require_existing_file(
                cfg.normalized_motion_file,
                "Normalized motion file",
            )
            checkpoint_file = require_existing_file(
                cfg.checkpoint_file,
                "Checkpoint file",
            )
            play_textop_motion(
                cfg,
                motion_file=motion_file,
                checkpoint_file=checkpoint_file,
            )
            return

        case EvalCommand():
            motion_file = require_existing_file(
                cfg.normalized_motion_file,
                "Normalized motion file",
            )
            checkpoint_file = require_existing_file(
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
