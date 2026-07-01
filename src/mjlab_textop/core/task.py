from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StaticTaskSpec:
    task_id: str
    make_env_cfg: Callable[[], Any]
    make_play_env_cfg: Callable[[], Any]
    make_rl_cfg: Callable[[], Any]
    runner_cls: type | None = None


def register_static_task_spec(spec: StaticTaskSpec) -> None:
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if spec.task_id in list_tasks():
        return

    register_mjlab_task(
        task_id=spec.task_id,
        env_cfg=spec.make_env_cfg(),
        play_env_cfg=spec.make_play_env_cfg(),
        rl_cfg=spec.make_rl_cfg(),
        runner_cls=spec.runner_cls,
    )


def register_static_task_specs(specs: Sequence[StaticTaskSpec]) -> None:
    for spec in specs:
        register_static_task_spec(spec)
