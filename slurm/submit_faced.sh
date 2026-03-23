#!/bin/bash
# Submit FACED downstream task jobs (LP and LP+FT) to the cluster.
#
# Usage:
#   bash slurm/submit_faced.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

echo "=== Submitting FACED Downstream Task Jobs ==="
echo ""

# 1. Linear probing (paper default — fast, ~20 epochs)
JOB1=$(sbatch --job-name="faced-lp" \
    slurm/run_dt.sh \
    training_mode=lp \
    'hydra.run.dir=outputs/faced_lp' \
    | awk '{print $4}')
echo "1/2  LP       job $JOB1  →  outputs/faced_lp/"

# 2. Linear probing + fine-tuning (LP warmup then full FT, ~220 epochs total)
JOB2=$(sbatch --job-name="faced-lpft" \
    slurm/run_dt.sh \
    training_mode=lp+ft \
    'hydra.run.dir=outputs/faced_lpft' \
    | awk '{print $4}')
echo "2/2  LP+FT    job $JOB2  →  outputs/faced_lpft/"

echo ""
echo "Monitor:  squeue --me"
echo "Logs:     slurm/logs/<job_id>.out"
echo ""
echo "After completion, sync W&B from login node:"
echo "  wandb sync /home/apasinato/reve-reproduce/src/outputs/faced_lp/wandb/latest-run"
echo "  wandb sync /home/apasinato/reve-reproduce/src/outputs/faced_lpft/wandb/latest-run"
