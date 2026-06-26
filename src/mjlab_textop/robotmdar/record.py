from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mjlab_textop.core.robotmdar import (
    robotmdar_motion_dict_to_block,
    slice_motion_dict_tail,
)
from mjlab_textop.core.robotmdar_record import save_robotmdar_raw_record
from mjlab_textop.robotmdar.produce import (
    _load_robotmdar_runtime,
    _register_hydra_resolvers,
)


def run_record(args: argparse.Namespace) -> None:
    runtime = _load_robotmdar_runtime()

    _register_hydra_resolvers(runtime.OmegaConf)
    cfg = runtime.OmegaConf.load(Path(args.ckpt).parent / ".hydra" / "config.yaml")
    cfg.device = args.device
    cfg.ckpt.dar = args.ckpt
    cfg.train.manager.device = args.device
    cfg.train.manager.save_dir = str(Path.cwd() / "logs" / "robotmdar_record")
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
    text_embedding = runtime.encode_text(clip_model, [args.prompt]).float()

    recorded_blocks = []
    frame_index = 0
    for block_index in range(args.num_blocks):
        block_start_time = time.monotonic()
        with runtime.torch.no_grad():
            future_motion, motion_dict, abs_pose = runtime.generate_next_motion(
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
        history_motion = future_motion[:, -history_len:, :]
        block = robotmdar_motion_dict_to_block(
            slice_motion_dict_tail(motion_dict, future_len),
            index=frame_index,
        )
        recorded_blocks.append(block)
        frame_index += block.joint_pos.shape[0]

        if args.log_every_blocks > 0 and (block_index + 1) % args.log_every_blocks == 0:
            generation_ms = (time.monotonic() - block_start_time) * 1000.0
            print(
                "record "
                f"block={block_index + 1}/{args.num_blocks} "
                f"frame={frame_index} prompt={args.prompt!r} "
                f"gen_ms={generation_ms:.1f}",
                file=sys.stderr,
            )

    save_robotmdar_raw_record(
        args.output,
        recorded_blocks,
        fps=args.fps,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
    )
    print(
        f"Recorded {len(recorded_blocks)} RobotMDAR blocks "
        f"({frame_index} frames) to {args.output}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a raw RobotMDAR reference record without MJLab live play.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--skeleton-asset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--prompt", default="walk")
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-every-blocks", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_blocks <= 0:
        raise ValueError(f"--num-blocks must be positive, got {args.num_blocks}")
    run_record(args)


if __name__ == "__main__":
    main()
