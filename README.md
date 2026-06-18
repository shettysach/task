# MJLab TextOp Playground

Utilities for running low-level TextOp tracker motions through MJLab's native
Unitree G1 tracking stack.

The current integration boundary is deliberately narrow: convert a canonical
TextOp tracker NPZ into MJLab's native tracking NPZ format, then use MJLab's
existing `MotionCommand` task, rewards, metrics, and play/train/evaluate flow.

## Architecture

```text
TextOp tracker NPZ
  -> normalize-textop-npz
  -> MJLab-native motion.npz
  -> Mjlab-Tracking-Flat-Unitree-G1
  -> MJLab MotionCommand
```

## Dependencies

This package uses upstream MJLab pinned to the latest verified `main` commit:

```text
0cdc56246999409b83622764f5b38edb660cf16e
```

It does not modify or depend on `../repos/mjlab`.

Dependency selection follows MJLab's upstream uv extras pattern:

```text
cpu   -> mjlab + torch from pytorch-cpu
cu128 -> mjlab + torch from pytorch-cu128
```

Use exactly one extra at a time. For local CPU verification:

```bash
uv sync --extra cpu
```

For the GPU machine:

```bash
uv sync --extra cu128
```

`pyproject.toml` declares the extras as conflicting, so uv rejects selecting
both CPU and CUDA dependencies in the same environment.

The extras depend on plain `mjlab`, not `mjlab[cpu]` or `mjlab[cu128]`. This
repo is the top-level uv project, so it owns the torch wheel selection through
`tool.uv.sources`. Pulling MJLab's own extras transitively causes uv to merge
CPU and CUDA torch indexes during lock resolution.

## Local shell

Use the Nix shell through `direnv` before running MJLab commands:

```bash
direnv reload
```

The shell defaults MuJoCo to EGL with Mesa `llvmpipe` so local verification can
run without CUDA:

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

On a GPU machine, keep the same Python code and use `--extra cu128`. If the
vendor EGL stack is available, override the shell variables or remove the Mesa
software-driver hints.

## Commands

For low-level TextOp tracker motions, normalize the TextOp NPZ into MJLab's
native tracking format first:

```bash
uv run --extra cpu normalize-textop-npz \
  --input-file /path/to/textop_motion.npz \
  --output-file /tmp/textop_mjlab_motion.npz \
  --device cpu
```

Then use MJLab's built-in G1 tracking task and `MotionCommand`:

```bash
uv run --extra cpu play Mjlab-Tracking-Flat-Unitree-G1 \
  --agent zero \
  --motion-file /tmp/textop_mjlab_motion.npz \
  --num-envs 1 \
  --no-terminations
```

The normalizer expects TextOp's canonical tracker NPZ fields. It reorders
TextOp IsaacLab G1 joints into MJLab/MuJoCo order and replays root plus joints
through MJLab so body references are written in MJLab's own body order.
