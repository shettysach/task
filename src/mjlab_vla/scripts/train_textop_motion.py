from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import tyro
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from mjlab_vla.scripts.eval_textop_motion import EvalCommand, evaluate_textop_motion
from mjlab_vla.scripts.normalize_textop_npz import normalize_textop_npz

DEFAULT_MOTION_REL = (
    "TextOpTracker/artifacts/Data10k-open/"
    "homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/"
    "motion.npz"
)


@dataclass
class NormalizedMotionConfig:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"


@dataclass
class NormalizeCommand(NormalizedMotionConfig):
    motion_rel: str = DEFAULT_MOTION_REL
    data_dir: str = "/tmp/textop-data"
    device: str = "cuda:0"


@dataclass
class TrainCommand(NormalizedMotionConfig):
    num_envs: int = 4096
    max_iterations: int = 10000
    logger: Literal["tensorboard", "wandb"] = "wandb"
    experiment_name: str = "textop_tracking"
    run_name: str = "walk_scratch"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


@dataclass
class PlayCommand(NormalizedMotionConfig):
    checkpoint_file: str | None = None
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"


TextOpCommand: TypeAlias = NormalizeCommand | TrainCommand | PlayCommand | EvalCommand

TextOpCommandType = tyro.extras.subcommand_type_from_defaults(
    {
        "normalize": NormalizeCommand(),
        "train": TrainCommand(),
        "play": PlayCommand(),
        "eval": EvalCommand(),
    },
)


def run_textop_motion(cfg: TextOpCommand) -> None:
    match cfg:
        case NormalizeCommand():
            normalized_file = Path(cfg.normalized_motion_file).expanduser()
            input_file = Path(cfg.data_dir).expanduser() / cfg.motion_rel
            normalize_textop_npz(
                input_file=str(input_file),
                output_file=str(normalized_file),
                device=cfg.device,
            )
            return

        case TrainCommand():
            normalized_file = Path(cfg.normalized_motion_file).expanduser()
            if not normalized_file.exists():
                raise FileNotFoundError(
                    f"Normalized motion file does not exist: {normalized_file}."
                )

            _train(
                motion_file=normalized_file,
                train_cfg=cfg,
            )
            return

        case PlayCommand():
            normalized_file = Path(cfg.normalized_motion_file).expanduser()
            if not normalized_file.exists():
                raise FileNotFoundError(
                    f"Normalized motion file does not exist: {normalized_file}."
                )

            if cfg.checkpoint_file is None:
                raise ValueError("`--checkpoint-file` is required for play")

            _play(
                motion_file=normalized_file,
                checkpoint_file=Path(cfg.checkpoint_file).expanduser(),
                play_cfg=cfg,
                device=cfg.device,
            )

        case EvalCommand():
            evaluate_textop_motion(cfg)
            return


def _train(
    motion_file: Path,
    train_cfg: TrainCommand,
) -> None:
    cfg = TrainConfig.from_task("Mjlab-Tracking-Flat-Unitree-G1")

    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_file)

    cfg.env.scene.num_envs = train_cfg.num_envs
    cfg.agent.max_iterations = train_cfg.max_iterations
    cfg.agent.logger = train_cfg.logger
    cfg.agent.experiment_name = train_cfg.experiment_name
    cfg.agent.run_name = train_cfg.run_name
    cfg.agent.resume = train_cfg.resume
    cfg.agent.load_run = train_cfg.load_run
    cfg.agent.load_checkpoint = train_cfg.load_checkpoint

    launch_training("Mjlab-Tracking-Flat-Unitree-G1", cfg)


def _play(
    motion_file: Path,
    checkpoint_file: Path,
    play_cfg: PlayCommand,
    device: str,
) -> None:
    cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(checkpoint_file),
        motion_file=str(motion_file),
        num_envs=play_cfg.num_envs,
        device=device,
        viewer=play_cfg.viewer,
    )
    run_play("Mjlab-Tracking-Flat-Unitree-G1", cfg)


def main() -> None:
    run_textop_motion(tyro.cli(TextOpCommandType))


if __name__ == "__main__":
    main()
