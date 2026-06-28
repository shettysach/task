# ty:ignore[unresolved-import]

from __future__ import annotations

import argparse
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mjlab_textop.core.online.live import textop_block_to_ndjson_message
from mjlab_textop.core.robotmdar import (
    robotmdar_motion_dict_to_block,
    slice_motion_dict_tail,
)
from mjlab_textop.robotmdar.feedback import UdpFeedbackReceiver
from mjlab_textop.robotmdar.planner import (
    ConstantPromptSelector,
    FeedbackPlanner,
    ManualPromptPlanner,
    OpenAIChatPromptSelector,
    PlannerContext,
    PromptPlanner,
)


@dataclass(frozen=True)
class RobotMdarRuntime:
    torch: Any
    OmegaConf: Any
    instantiate: Callable[..., Any]
    seed: Any
    ClassifierFreeWrapper: type
    generate_next_motion: Callable[..., Any]
    load_and_freeze_clip: Callable[..., Any]
    encode_text: Callable[..., Any]
    get_zero_abs_pose: Callable[..., Any]
    get_zero_feature: Callable[..., Any]


def run_producer(args: argparse.Namespace) -> None:
    runtime = _load_robotmdar_runtime()

    _register_hydra_resolvers(runtime.OmegaConf)
    cfg = runtime.OmegaConf.load(Path(args.ckpt).parent / ".hydra" / "config.yaml")
    cfg.device = args.device
    cfg.ckpt.dar = args.ckpt
    cfg.train.manager.device = args.device
    cfg.train.manager.save_dir = str(Path.cwd() / "logs" / "robotmdar_producer")
    cfg.train.manager.platform._target_ = "robotmdar.train.train_platforms.NoPlatform"
    cfg.data.datadir = args.datadir
    cfg.skeleton.asset.assetRoot = args.skeleton_asset_root
    cfg.data.val.split = "none"
    cfg.data.val.batch_size = 1
    cfg.use_full_sample = True
    cfg.guidance_scale = args.guidance_scale

    runtime.seed.set(cfg.seed)
    clip_model = runtime.load_and_freeze_clip("ViT-B/32", device=args.device)
    val_data = runtime.instantiate(cfg.data.val)
    vae = runtime.instantiate(cfg.vae)
    denoiser = runtime.instantiate(cfg.denoiser)
    schedule_sampler = runtime.instantiate(cfg.diffusion.schedule_sampler)
    diffusion = schedule_sampler.diffusion
    vae.eval()
    denoiser.eval()

    manager = runtime.instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)
    cfg_denoiser = runtime.ClassifierFreeWrapper(denoiser)

    future_len = int(cfg.data.future_len)
    history_len = int(cfg.data.history_len)
    history_motion = val_data.normalize(
        runtime.get_zero_feature()
        .to(args.device)
        .reshape(1, 1, -1)
        .repeat(1, history_len, 1)
    )
    abs_pose = runtime.get_zero_abs_pose((1,), device=args.device)
    planner = make_prompt_planner(args)
    planner.start()
    if isinstance(planner, FeedbackPlanner):
        print("Using feedback planner.", file=sys.stderr)

    try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((args.host, args.port))
                server.listen(1)
                print(f"Waiting for MJLab consumer on {args.host}:{args.port}")
                conn, addr = server.accept()
                print(f"MJLab consumer connected from {addr}")
                with conn:
                    frame_index = 0
                    next_send_time = time.monotonic()
                    block_count = 0
                    while not planner.should_stop:
                        block_start_time = time.monotonic()
                        current_prompt = planner.choose_prompt(
                            PlannerContext(
                                frame_index=frame_index,
                                block_count=block_count,
                            )
                        )
                        with runtime.torch.no_grad():
                            text_embedding = runtime.encode_text(
                                clip_model, [current_prompt]
                            ).float()
                            future_motion, motion_dict, abs_pose = (
                                runtime.generate_next_motion(
                                    vae=vae,
                                    denoiser=cfg_denoiser,
                                    diffusion=diffusion,
                                    val_data=val_data,
                                    text_embedding=text_embedding,
                                    history_motion=history_motion,
                                    abs_pose=abs_pose,
                                    future_len=future_len,
                                    use_full_sample=True,
                                    guidance_scale=args.guidance_scale,
                                    ret_fk=True,
                                    ret_fk_full=False,
                                )
                            )
                        history_motion = future_motion[:, -history_len:, :]
                        block = robotmdar_motion_dict_to_block(
                            slice_motion_dict_tail(motion_dict, future_len),
                            index=frame_index,
                        )
                        conn.sendall(
                            textop_block_to_ndjson_message(block, fps=args.fps).encode(
                                "utf-8"
                            )
                        )
                        block_frames = block.joint_pos.shape[0]
                        frame_index += block_frames
                        block_count += 1

                        block_duration = block_frames / args.fps
                        next_send_time += block_duration
                        sleep_seconds = next_send_time - time.monotonic()
                        if (
                            args.log_every_blocks > 0
                            and block_count % args.log_every_blocks == 0
                            and not planner.input_active
                        ):
                            generation_ms = (
                                time.monotonic() - block_start_time
                            ) * 1000.0
                            lag_ms = max(0.0, -sleep_seconds * 1000.0)
                            print(
                                "stream "
                                f"block={block_count} frame={frame_index} "
                                f"prompt={current_prompt!r} "
                                f"gen_ms={generation_ms:.1f} "
                                f"lag_ms={lag_ms:.1f}"
                                f"{planner.log_suffix}",
                                file=sys.stderr,
                                end="",
                                flush=True,
                            )
                        time.sleep(max(0.0, sleep_seconds))
        finally:
            planner.request_stop()
    except KeyboardInterrupt:
        planner.request_stop()
        print("Stopping RobotMDAR producer.", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream RobotMDAR text-to-motion blocks to MJLab over NDJSON TCP.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--skeleton-asset-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--planner",
        choices=("manual", "feedback", "vlm"),
        default="manual",
    )
    parser.add_argument("--prompt", default="walk")
    parser.add_argument("--feedback-listen-host", default="127.0.0.1")
    parser.add_argument("--feedback-listen-port", type=int, default=None)
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:9379")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-timeout-sec", type=float, default=2.0)
    parser.add_argument("--vlm-max-completion-tokens", type=int, default=32)
    parser.add_argument("--query-every-blocks", type=int, default=4)
    parser.add_argument("--fallback-prompt", default="stand still")
    parser.add_argument("--stale-steps-threshold", type=int, default=5)
    parser.add_argument("--fall-recovery-blocks", type=int, default=8)
    parser.add_argument("--feedback-timeout-sec", type=float, default=None)
    parser.add_argument("--log-every-blocks", type=int, default=20)
    args = parser.parse_args()
    if args.planner in {"feedback", "vlm"} and args.feedback_listen_port is None:
        raise ValueError(
            f"--feedback-listen-port is required with --planner {args.planner}"
        )
    if args.query_every_blocks <= 0:
        raise ValueError(
            f"--query-every-blocks must be positive, got {args.query_every_blocks}"
        )
    if args.stale_steps_threshold < 0:
        raise ValueError(
            "--stale-steps-threshold must be non-negative, "
            f"got {args.stale_steps_threshold}"
        )
    if args.fall_recovery_blocks < 0:
        raise ValueError(
            "--fall-recovery-blocks must be non-negative, "
            f"got {args.fall_recovery_blocks}"
        )
    if args.vlm_timeout_sec <= 0:
        raise ValueError(
            f"--vlm-timeout-sec must be positive, got {args.vlm_timeout_sec}"
        )
    if args.vlm_max_completion_tokens <= 0:
        raise ValueError(
            "--vlm-max-completion-tokens must be positive, "
            f"got {args.vlm_max_completion_tokens}"
        )
    if args.planner == "vlm" and not args.vlm_model:
        raise ValueError("--vlm-model is required with --planner vlm")
    return args


def main() -> None:
    run_producer(parse_args())


def _load_robotmdar_runtime() -> RobotMdarRuntime:
    try:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from robotmdar.dtype import seed
        from robotmdar.dtype.motion import get_zero_abs_pose, get_zero_feature
        from robotmdar.eval.generate_dar import (
            ClassifierFreeWrapper,
            generate_next_motion,
        )
        from robotmdar.model.clip import encode_text, load_and_freeze_clip
    except ImportError as exc:
        raise ImportError(
            "RobotMDAR producer must be run in the TextOp/RobotMDAR environment."
        ) from exc
    return RobotMdarRuntime(
        torch=torch,
        instantiate=instantiate,
        OmegaConf=OmegaConf,
        seed=seed,
        get_zero_abs_pose=get_zero_abs_pose,
        get_zero_feature=get_zero_feature,
        ClassifierFreeWrapper=ClassifierFreeWrapper,
        generate_next_motion=generate_next_motion,
        encode_text=encode_text,
        load_and_freeze_clip=load_and_freeze_clip,
    )


def _register_hydra_resolvers(OmegaConf) -> None:
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver(
            "hydra",
            lambda key: str(Path.cwd()) if key == "runtime.cwd" else "",
        )
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda fmt: datetime.now().strftime(fmt),
        )


def make_prompt_planner(args: argparse.Namespace) -> PromptPlanner:
    if args.planner in {"feedback", "vlm"}:
        receiver = UdpFeedbackReceiver(
            host=args.feedback_listen_host,
            port=args.feedback_listen_port,
        )
        selector = (
            OpenAIChatPromptSelector(
                base_url=args.vlm_base_url,
                model=args.vlm_model,
                timeout_sec=args.vlm_timeout_sec,
                max_completion_tokens=args.vlm_max_completion_tokens,
            )
            if args.planner == "vlm"
            else ConstantPromptSelector(args.prompt)
        )
        return FeedbackPlanner(
            observation_provider=receiver,
            selector=selector,
            initial_prompt=args.prompt,
            query_every_blocks=args.query_every_blocks,
            fallback_prompt=args.fallback_prompt,
            stale_steps_threshold=args.stale_steps_threshold,
            fall_recovery_blocks=args.fall_recovery_blocks,
            feedback_timeout_sec=args.feedback_timeout_sec,
        )
    return ManualPromptPlanner(args.prompt)


if __name__ == "__main__":
    main()
