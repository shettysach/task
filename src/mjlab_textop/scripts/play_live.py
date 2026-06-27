from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
)
from mjlab_textop.core.online.live_registry import (
    register_live_textop_source,
    unregister_live_textop_source,
)
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import (
    ensure_textop_task_registered,
    register_online_textop_onnx_task,
    register_online_textop_task,
)
from mjlab_textop.scripts.policy import ResolvedPolicy, run_textop_play


@dataclass(kw_only=True)
class PlayLiveCommand:
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    fps: float = 50.0
    max_queue_blocks: int = 32
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    ensure_textop_task_registered()
    source = SocketTextOpOnlineSource(
        SocketTextOpSourceCfg(
            host=cfg.host,
            port=cfg.port,
            fps=cfg.fps,
            max_queue_blocks=cfg.max_queue_blocks,
        )
    )
    source.start()
    source_key = register_live_textop_source(source)
    try:
        if policy.kind == "onnx":
            task_name = register_online_textop_onnx_task(
                source_key=source_key,
                source_mode="live",
                future_steps=cfg.future_steps,
                num_envs=cfg.num_envs,
                anchor_alignment=cfg.anchor_alignment,
            )
        else:
            task_name = register_online_textop_task(
                source_key=source_key,
                source_mode="live",
                future_steps=cfg.future_steps,
                num_envs=cfg.num_envs,
                anchor_alignment=cfg.anchor_alignment,
            )

        run_textop_play(task_name, policy.file, cfg)
    finally:
        unregister_live_textop_source(source_key)
        source.close()
