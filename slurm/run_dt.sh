#!/bin/bash
# Generic single-GPU SLURM job script for REVE downstream task on FACED.
#
# Usage:
#   sbatch slurm/run_dt.sh [hydra overrides...]
#
# Examples:
#   # Linear probing (paper default)
#   sbatch slurm/run_dt.sh training_mode=lp 'hydra.run.dir=outputs/faced_lp'
#
#   # Linear probing + fine-tuning
#   sbatch slurm/run_dt.sh training_mode=lp+ft 'hydra.run.dir=outputs/faced_lpft'
#
#   # Fine-tuning only
#   sbatch slurm/run_dt.sh training_mode=ft 'hydra.run.dir=outputs/faced_ft'

#SBATCH --job-name="reve-faced-dt"
#SBATCH --partition=gpu-a100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/apasinato/reve-reproduce
SCRATCH=/scratch/apasinato
DATA_ROOT=$SCRATCH/data
MODELS_ROOT=$SCRATCH/models

# ── No internet on compute nodes — load models from scratch ───────────────────
export REVE_POSITIONS_MODEL=$MODELS_ROOT/reve-positions

# ── W&B offline (sync from login node after job completes) ────────────────────
export WANDB_MODE=offline
[ -f $REPO/.env ] && export $(grep -v '^#' $REPO/.env | xargs)

# ── Info ──────────────────────────────────────────────────────────────────────
echo "=== REVE Downstream Task — FACED ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Args:      $@"
echo "Started:   $(date)"
echo ""

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Environment ───────────────────────────────────────────────────────────────
cd $REPO

module load 2024r1
module load cuda/12.5

export PATH="$HOME/.local/bin:$PATH"

echo "=== Environment ==="
echo "Python:  $(uv run python --version)"
echo "PyTorch: $(uv run python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(uv run python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== Running dt.py ==="
echo ""

cd $REPO/src

uv run python dt.py \
    task=faced \
    data_root=$DATA_ROOT \
    pretrained_path=hf:$MODELS_ROOT/reve-base \
    loader.num_workers=8 \
    "$@"

EXIT_CODE=$?

echo ""
echo "=== Complete ==="
echo "Exit code: $EXIT_CODE"
echo "Finished:  $(date)"
echo ""
echo "Sync W&B from login node:"
echo "  wandb sync $REPO/src/outputs/faced_*/wandb/latest-run"

exit $EXIT_CODE
