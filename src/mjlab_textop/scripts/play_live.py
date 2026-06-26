from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.contract import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
)
from mjlab_textop.core.online.live_registry import (
    register_live_textop_source,
    unregister_live_textop_source,
)
from mjlab_textop.core.robotmdar import save_textop_motion_blocks_as_mjlab_npz
from mjlab_textop.core.task import (
    ensure_textop_task_registered,
    register_online_textop_task,
)


@dataclass(kw_only=True)
class PlayLiveCommand:
    checkpoint_file: str = field(default=tyro.MISSING)
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    fps: float = 50.0
    max_queue_blocks: int = 32
    max_stale_steps: int = 25
    log_metrics_every_steps: int = 0
    record_output: str | None = None
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    checkpoint_file: Path,
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
        task_name = register_online_textop_task(
            source_key=source_key,
            source_mode="live",
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
            max_stale_steps=cfg.max_stale_steps,
            log_metrics_every_steps=cfg.log_metrics_every_steps,
        )

        play_cfg = PlayConfig(
            agent="trained",
            checkpoint_file=str(checkpoint_file),
            num_envs=cfg.num_envs,
            device=cfg.device,
            viewer=cfg.viewer,
        )
        run_play(task_name, play_cfg)
    finally:
        unregister_live_textop_source(source_key)
        source.close()
        if cfg.record_output is not None:
            _save_recorded_live_stream(
                cfg.record_output,
                source=source,
            )


def _save_recorded_live_stream(
    record_output: str,
    *,
    source: SocketTextOpOnlineSource,
) -> None:
    output_path = Path(record_output).expanduser().resolve()
    blocks = source.recorded_blocks()
    if not blocks:
        print(
            f"No live TextOp blocks received; skipping record output {output_path}",
            file=sys.stderr,
        )
        return
    save_textop_motion_blocks_as_mjlab_npz(
        output_path,
        blocks,
        fps=source.fps,
    )
    print(
        f"Recorded {len(blocks)} live TextOp block(s) to {output_path}",
        file=sys.stderr,
    )
