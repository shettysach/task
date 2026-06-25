# MJLab TextOp

Utilities for running low-level TextOp tracker motions through MJLab's native
Unitree G1 tracking stack.

The integration supports three paths:

- offline tracking: convert a canonical TextOp tracker NPZ into MJLab's native
  tracking NPZ format, then train/play/evaluate a TextOp-style MJLab task.
- RobotMDAR dataset generation: generate a raw RobotMDAR reference record
  without MJLab live play, then normalize it into a train-ready MJLab NPZ.
- online replay: stream a normalized MJLab motion file through the same online
  reference buffer used by live sources.
- live text-to-motion demo: run RobotMDAR in a separate TextOp environment and
  stream generated TextOp motion blocks to MJLab over localhost NDJSON.

## Architecture

```text
TextOp tracker NPZ
  -> mjlab-textop normalize-tracker-npz
  -> MJLab-native motion.npz
  -> Mjlab-TextOp-Flat-Unitree-G1
  -> TextOpMotionCommand
  -> TextOp-style future reference observations

RobotMDAR text prompt
  -> mjlab-textop-robotmdar
  -> localhost NDJSON TextOpMotionBlock stream
  -> mjlab-textop play-live
  -> OnlineTextOpMotionCommand
  -> TextOp-style future reference observations

RobotMDAR text prompt
  -> mjlab_textop.scripts.robotmdar_record
  -> raw RobotMDAR reference NPZ
  -> mjlab-textop normalize-robotmdar-npz
  -> MJLab-native train-ready motion.npz
  -> Mjlab-TextOp-Flat-Unitree-G1
```

The reusable TextOp integration code lives under `src/mjlab_textop/core/`.
Installed command-line entry points live under `src/mjlab_textop/scripts/` and reuse
the library modules.

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

### `normalize-tracker-npz`

Download a TextOp tracker NPZ and convert it into MJLab's native tracking format:

```bash
uvx hf download Yochish/TextOp-Data \
  --repo-type dataset \
  --include 'TextOpTracker/artifacts/Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/motion.npz' \
  --local-dir /tmp/textop-data

uv run --extra cu128 mjlab-textop normalize-tracker-npz \
  --motion-file /tmp/textop-data/TextOpTracker/artifacts/Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/motion.npz \
  --normalized-motion-file ./outputs/walk_mjlab.npz
```

The normalizer expects TextOp's canonical tracker NPZ fields. It reorders
TextOp IsaacLab G1 joints into MJLab/MuJoCo order and replays root plus joints
through MJLab so body references are written in MJLab's own body order.

Offline TextOp tracking uses `torso_link` as the configured anchor body. Online
streaming uses `pelvis`, matching the root/anchor term carried by live
`TextOpMotionBlock` messages.

### `robotmdar-record`

Generate a raw RobotMDAR reference record without starting an MJLab socket
consumer. Run this in the TextOp/RobotMDAR Python environment with this package
on `PYTHONPATH`:

```bash
uv run python -m mjlab_textop.scripts.robotmdar_record \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --prompt "walk forward" \
  --num-blocks 200 \
  --output ./outputs/robotmdar_walk_raw.npz
```

The raw record stores `joint_pos`, `joint_vel`, `anchor_pos_w`, and
`anchor_quat_w`. Joint arrays remain in TextOp/IsaacLab G1 order.

### `normalize-robotmdar-npz`

Convert a raw RobotMDAR record into the full MJLab train-ready NPZ. Run this in
the MJLab environment:

```bash
uv run --extra cu128 mjlab-textop normalize-robotmdar-npz \
  --recorded-motion-file ./outputs/robotmdar_walk_raw.npz \
  --normalized-motion-file ./outputs/robotmdar_walk_train_ready.npz
```

This command reindexes RobotMDAR raw joints from TextOp/IsaacLab order into
MJLab order exactly once, uses the raw anchor trajectory as the robot root, runs
MJLab forward kinematics, and saves full MJLab body position, orientation, and
velocity arrays.

### `train`

Train the TextOp tracking task on the normalized motion. Checkpoints are
saved to `logs/rsl_rl/textop_tracking/<timestamp>_walk_scratch/model_<iter>.pt`.

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/walk_mjlab.npz \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.experiment-name textop_tracking \
  --agent.run-name walk_scratch

# Checkpoint saved to:
#   logs/rsl_rl/textop_tracking/<timestamp>_walk_scratch/model_<iteration>.pt
export CHECKPOINT=$(ls -t logs/rsl_rl/textop_tracking/*_walk_scratch/model_*.pt | head -1)
```

To finetune from a previous run:

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/walk_mjlab.npz \
  --agent.resume True \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.experiment-name textop_tracking \
  --agent.load-run '.*walk_scratch.*' \
  --agent.load-checkpoint 'model_.*.pt' \
  --agent.run-name walk_finetune
```

### `play`

View a trained checkpoint with the native MJLab viewer:

```bash
uv run --extra cu128 play Mjlab-TextOp-Flat-Unitree-G1 \
  --checkpoint-file $CHECKPOINT \
  --motion-file ./outputs/walk_mjlab.npz
```

### `play-online`

Replay the normalized motion through the online TextOp reference buffer:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --checkpoint-file $CHECKPOINT \
  --motion-file ./outputs/walk_mjlab.npz
```

### `play-live`

Run a live text-to-motion demo over localhost NDJSON. Start the RobotMDAR
producer in the TextOp/RobotMDAR Python environment:

#### Setup

```bash
cd .. # Don't clone within the mjlab_textop repo
git clone --recurse-submodules https://github.com/TeleHuman/TextOp.git
cd TextOp

uv venv --python 3.10

uv pip install torch
uv pip install -e ./deps/isaac_utils
uv pip install git+https://github.com/openai/CLIP.git
uv pip install -e ./TextOpRobotMDAR

# Download
uvx hf download Yochish/TextOp-Data \
  --repo-type dataset \
  --local-dir /tmp/textop-data \
  --include 'TextOpRobotMDAR/logs/**' \
  --include 'TextOpRobotMDAR/dataset/**' \
  --include 'TextOpRobotMDAR/description/**'

export PYTHONPATH="../mjlab_textop/src:$PYTHONPATH"

uv run python -m mjlab_textop.scripts.robotmdar_producer \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1
```

Then run MJLab in this repo's environment:

```bash
uv run --extra cu128 mjlab-textop play-live \
  --checkpoint-file $CHECKPOINT \
  --host 127.0.0.1 \
  --port 8765
```

The live producer sends 50 Hz-indexed motion chunks. MJLab consumes them at the
online command rate, clamps stale future frames during underruns, and reports
online buffer/source diagnostics through command metrics.

### `eval`

Run a headless evaluation against the normalized motion:

```bash
uv run --extra cu128 mjlab-textop eval \
  --checkpoint-file $CHECKPOINT \
  --motion-file ./outputs/walk_mjlab.npz \
  --num-envs 1024 \
  --output-file logs/textop_eval.json
```

Evaluation reuses MJLab's tracking metrics: success rate, global MPKPE,
root-relative MPKPE, joint velocity error, and end-effector pose errors.
