from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from mjlab_vla.textop.online.live import textop_block_to_ndjson_message
from mjlab_vla.textop.online.source import TextOpMotionBlock

MUJOCO_TO_ISAACLAB_REINDEX = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)


@dataclass
class PromptState:
    text: str = "stand"
    stop: bool = False


def expand_dof_23_to_29(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 23:
        raise ValueError(f"Expected [T, 23] RobotMDAR DoF array, got {value.shape}")
    out = np.zeros((value.shape[0], 29), dtype=np.float32)
    out[:, :19] = value[:, :19]
    out[:, 22:26] = value[:, 19:23]
    return out


def robotmdar_motion_dict_to_block(motion_dict, index: int) -> TextOpMotionBlock:
    joint_pos_mj = expand_dof_23_to_29(_to_numpy(motion_dict["dof_pos"][0]))
    joint_vel_mj = expand_dof_23_to_29(_to_numpy(motion_dict["dof_vel"][0]))
    root_rot_xyzw = _to_numpy(motion_dict["root_rot"][0])
    return TextOpMotionBlock(
        index=index,
        joint_pos=joint_pos_mj[:, MUJOCO_TO_ISAACLAB_REINDEX],
        joint_vel=joint_vel_mj[:, MUJOCO_TO_ISAACLAB_REINDEX],
        anchor_pos_w=_to_numpy(motion_dict["root_trans_offset"][0]),
        anchor_quat_w=root_rot_xyzw[:, [3, 0, 1, 2]],
    )


def run_producer(args: argparse.Namespace) -> None:
    imports = _load_robotmdar_imports()
    torch = imports["torch"]
    OmegaConf = imports["OmegaConf"]
    instantiate = imports["instantiate"]
    seed = imports["seed"]
    ClassifierFreeWrapper = imports["ClassifierFreeWrapper"]
    generate_next_motion = imports["generate_next_motion"]
    load_and_freeze_clip = imports["load_and_freeze_clip"]
    encode_text = imports["encode_text"]
    get_zero_abs_pose = imports["get_zero_abs_pose"]
    get_zero_feature = imports["get_zero_feature"]

    _register_hydra_resolvers(OmegaConf)
    cfg = OmegaConf.load(Path(args.ckpt).parent / ".hydra" / "config.yaml")
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

    seed.set(cfg.seed)
    clip_model = load_and_freeze_clip("ViT-B/32", device=args.device)
    val_data = instantiate(cfg.data.val)
    vae = instantiate(cfg.vae)
    denoiser = instantiate(cfg.denoiser)
    schedule_sampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion = schedule_sampler.diffusion
    vae.eval()
    denoiser.eval()

    manager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)
    cfg_denoiser = ClassifierFreeWrapper(denoiser)

    future_len = int(cfg.data.future_len)
    history_len = int(cfg.data.history_len)
    history_motion = val_data.normalize(
        get_zero_feature().to(args.device).reshape(1, 1, -1).repeat(
            1,
            history_len,
            1,
        )
    )
    abs_pose = get_zero_abs_pose((1,), device=args.device)
    prompt = PromptState(text=args.prompt)
    input_thread = threading.Thread(target=_prompt_loop, args=(prompt,), daemon=True)
    input_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Waiting for MJLab consumer on {args.host}:{args.port}")
        conn, addr = server.accept()
        print(f"MJLab consumer connected from {addr}")
        with conn:
            frame_index = 0
            while not prompt.stop:
                with torch.no_grad():
                    text_embedding = encode_text(clip_model, [prompt.text]).float()
                    future_motion, motion_dict, abs_pose = generate_next_motion(
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
                    _slice_motion_dict_tail(motion_dict, future_len),
                    frame_index,
                )
                conn.sendall(
                    textop_block_to_ndjson_message(block, fps=args.fps).encode("utf-8")
                )
                frame_index += block.joint_pos.shape[0]
                time.sleep(max(0.0, block.joint_pos.shape[0] / args.fps))


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
    parser.add_argument("--prompt", default="stand")
    return parser.parse_args()


def main() -> None:
    run_producer(parse_args())


def _load_robotmdar_imports() -> dict[str, object]:
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
    return {
        "torch": torch,
        "instantiate": instantiate,
        "OmegaConf": OmegaConf,
        "seed": seed,
        "get_zero_abs_pose": get_zero_abs_pose,
        "get_zero_feature": get_zero_feature,
        "ClassifierFreeWrapper": ClassifierFreeWrapper,
        "generate_next_motion": generate_next_motion,
        "encode_text": encode_text,
        "load_and_freeze_clip": load_and_freeze_clip,
    }


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


def _prompt_loop(prompt: PromptState) -> None:
    while not prompt.stop:
        try:
            text = input("Enter text prompt (or q to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            prompt.stop = True
            return
        if text.lower() in {"q", "quit", "exit"}:
            prompt.stop = True
        elif text:
            prompt.text = text


def _slice_motion_dict_tail(motion_dict, frames: int):
    result = {}
    for key, value in motion_dict.items():
        if hasattr(value, "shape") and len(value.shape) >= 2:
            result[key] = value[:, -frames:]
        else:
            result[key] = value
    return result


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


if __name__ == "__main__":
    main()
