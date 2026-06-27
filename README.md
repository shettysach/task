# MJLab TextOp

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

---

## Commands

### TextOpRobotMDAR

#### `robotmdar-record`

Generate a raw RobotMDAR reference record without starting an MJLab socket consumer. 
Run this in the TextOp/RobotMDAR Python environment with this package on `PYTHONPATH`:

##### TextOpRobotMDAR Setup

```bash
cd .. # Don't clone within the mjlab_textop repo
git clone --recurse-submodules https://github.com/TeleHuman/TextOp.git
cd TextOp

uv venv --python 3.10

uv pip install torch
uv pip install -e ./deps/isaac_utils
uv pip install git+https://github.com/openai/CLIP.git
uv pip install -e ./TextOpRobotMDAR

uvx hf download Yochish/TextOp-Data \
  --repo-type dataset \
  --local-dir /tmp/textop-data \
  --include 'TextOpRobotMDAR/logs/**' \
  --include 'TextOpRobotMDAR/dataset/**' \
  --include 'TextOpRobotMDAR/description/**'
```

```bash
# In TextOp directory
uv run python -m mjlab_textop.robotmdar.record \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --prompt "walk forward" \
  --num-blocks 200 \
  --output /tmp/walk_forward.npz
```

The raw record stores `joint_pos`, `joint_vel`, `anchor_pos_w`, and
`anchor_quat_w`. Joint arrays remain in TextOp/IsaacLab G1 order.

#### `normalize`

Convert a raw RobotMDAR record into the full MJLab train-ready NPZ. Run this in
the MJLab environment:

```bash
uv run --extra cu128 mjlab-textop normalize \
  --input-motion-file /tmp/walk_forward.npz \
  --output-motion-file ./outputs/walk_forward.npz
```

This command reindexes RobotMDAR raw joints from TextOp/IsaacLab order into
MJLab order exactly once, uses the raw anchor trajectory as the robot root, runs
MJLab forward kinematics, and saves full MJLab body position, orientation, and
velocity arrays.

#### `train`

Train the TextOp tracking task on the normalized motion using MJLab's `train` command: 

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/walk_forward.npz \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.experiment-name textop_tracking \
  --agent.run-name robotmdar_walk_forward \
  --env.commands.motion.anchor-body-name pelvis
```

```bash
# Checkpoint saved to: logs/rsl_rl/textop_tracking/<timestamp>_robotmdar_walk_forward/model_<iteration>.pt
export CHECKPOINT=logs/rsl_rl/textop_tracking/2026-06-25_00-20-00_robotmdar_walk_forward/model_5000.pt
```

To finetune from a previous run:

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/stand_still.npz \
  --agent.resume True \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.experiment-name textop_tracking \
  --agent.load-run 2026-06-25_00-20-00_robotmdar_pelvis_scratch \
  --agent.load-checkpoint model_5000.pt \
  --agent.run-name stand_still \
  --env.commands.motion.anchor-body-name pelvis
```

#### `play`

View a trained checkpoint using MJLab's `play` command: 

```bash
uv run --extra cu128 play Mjlab-TextOp-Flat-Unitree-G1 \
  --checkpoint-file $CHECKPOINT \
  --motion-file ./outputs/walk_forward.npz
```

#### `play-online`

Replay the normalized motion through the online TextOp reference buffer:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --checkpoint-file $CHECKPOINT \
  --motion-file ./outputs/walk_forward.npz
```

To replay with TextOp's released `latest.onnx` policy instead:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --onnx-file $ONNX_PATH \
  --motion-file ./outputs/walk_forward.npz
```

#### ONNX Setup

Use this setup before running a `play-*` command with `--onnx-file`:

```bash
uvx hf download Yochish/TextOp-Data \
TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx \
--repo-type dataset \
--local-dir /tmp

$ONNX_PATH=/tmp/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx
```

The `--checkpoint-file` and `--onnx-file` options are mutually exclusive.

#### `play-live`

Run a live text-to-motion demo over localhost NDJSON. 
Start the RobotMDAR producer in the TextOp/RobotMDAR Python environment:

Setup - [TextOpRobotMDAR Setup](#textoprobotmdar-setup)

```bash
# In TextOp directory
uv run python -m mjlab_textop.robotmdar.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1
```

```bash
# In mjlab_textop directory
uv run --extra cu128 mjlab-textop play-live \
  --checkpoint-file $CHECKPOINT \
  --host 127.0.0.1 \
  --port 8765
```

The live producer sends 50 Hz-indexed motion chunks. MJLab consumes them at the
online command rate, clamps stale future frames during underruns, and reports
online buffer/source diagnostics through command metrics.

To run the same live source with TextOp's released `latest.onnx` policy:

Setup - [ONNX Setup](#onnx-setup)

```bash
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file $ONNX_PATH \
  --host 127.0.0.1 \
  --port 8765
```

The ONNX path uses the online source and the ONNX actor directly, without a
`.pt` checkpoint.
