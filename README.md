# MJLab VLA

Utilities for running low-level TextOp tracker motions through MJLab's native
Unitree G1 tracking stack.

The current integration boundary is deliberately narrow: convert a canonical
TextOp tracker NPZ into MJLab's native tracking NPZ format, then train or play a
registered MJLab task variant with TextOp-style low-level tracker observations.
This is offline TextOp tracker integration; it does not run RobotMDAR or live
text-to-motion generation yet.

## Architecture

```text
TextOp tracker NPZ
  -> textop-tracking normalize
  -> MJLab-native motion.npz
  -> Mjlab-TextOp-Flat-Unitree-G1
  -> TextOpMotionCommand
  -> TextOp-style future reference observations
```

The TextOp integration code lives under `src/mjlab_vla/textop/`:
`contract.py`, `mdp/`, `task.py`, and the command-line scripts in `script/`.

## Dependencies

This package uses upstream MJLab pinned to the latest verified `main` commit.
Dependency selection follows MJLab's upstream uv extras pattern:

```text
cpu   -> mjlab + torch from pytorch-cpu
cu128 -> mjlab + torch from pytorch-cu128
```

Use exactly one extra at a time. For local CPU verification `pyproject.toml` declares the extras as conflicting, so uv rejects selecting
both CPU and CUDA dependencies in the same environment.

The extras depend on plain `mjlab`, not `mjlab[cpu]` or `mjlab[cu128]`. This
repo is the top-level uv project, so it owns the torch wheel selection through
`tool.uv.sources`. Pulling MJLab's own extras transitively causes uv to merge
CPU and CUDA torch indexes during lock resolution.

## Commands

For low-level TextOp tracker motions, download the TextOp NPZ explicitly:

```bash
uvx hf download Yochish/TextOp-Data \
  --repo-type dataset \
  --include 'TextOpTracker/artifacts/Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/motion.npz' \
  --local-dir /tmp/textop-data
```

Then normalize the TextOp NPZ into MJLab's native tracking format:

```bash
uv run --extra cu128 textop-tracking normalize
```

Then use the registered TextOp-flavored MJLab task:

```bash
uv run --extra cpu play Mjlab-TextOp-Flat-Unitree-G1 \
  --agent zero \
  --motion-file /tmp/textop_walk_mjlab.npz \
  --num-envs 1 \
  --no-terminations True
```

The normalizer expects TextOp's canonical tracker NPZ fields. It reorders
TextOp IsaacLab G1 joints into MJLab/MuJoCo order and replays root plus joints
through MJLab so body references are written in MJLab's own body order.

For the downloaded and normalized TextOp walking motion on a GPU machine, train
a TextOp-style MJLab tracking policy from scratch:

```bash
uv run --extra cu128 textop-tracking train
```

Useful overrides:

```bash
uv run --extra cu128 textop-tracking train \
  --num-envs 8192 \
  --max-iterations 30000 \
  --run-name walk_scratch_long
```

To finetune from a previous MJLab run:

```bash
uv run --extra cu128 textop-tracking train \
  --resume \
  --load-run '.*walk_scratch.*' \
  --load-checkpoint 'model_.*.pt' \
  --run-name walk_finetune
```

To view a trained MJLab checkpoint:

```bash
uv run --extra cu128 textop-tracking play \
  --checkpoint-file /path/to/model.pt
```

To run a headless local evaluation against the normalized TextOp motion:

```bash
uv run --extra cu128 textop-tracking eval \
  --checkpoint-file /path/to/model.pt \
  --num-envs 1024 \
  --output-file logs/textop_eval.json
```

Evaluation reuses MJLab's tracking metrics: success rate, global MPKPE,
root-relative MPKPE, joint velocity error, and end-effector pose errors.

To normalize a different downloaded TextOp motion:

```bash
uv run --extra cu128 textop-tracking normalize \
  --motion-rel path/inside/textop-data/motion.npz \
  --normalized-motion-file /tmp/other_textop_mjlab.npz
```
