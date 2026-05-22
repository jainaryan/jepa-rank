#!/usr/bin/env bash
#SBATCH --job-name=jepa-rank-exp2-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=xgpj0,xgpk0
#SBATCH --output=/home/a/aryanj/logs/jepa-rank-exp2-probe-%j.out
#SBATCH --error=/home/a/aryanj/logs/jepa-rank-exp2-probe-%j.err

set -euo pipefail

PROJ=$HOME/projects/ijepa
SRC_RUN_ID="${SRC_RUN_ID:-618949}"
SRC="$PROJ/outputs/exp2_lowrank_probe/${SRC_RUN_ID}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUT="$PROJ/outputs/exp2_lowrank_probe/${RUN_ID}"
mkdir -p "$OUT" "$HOME/logs"

# Reuse the extract output from the cancelled run.
ln -sfn "$SRC/features" "$OUT/features"

source "$PROJ/ijepaenv/bin/activate"
cd "$PROJ"

python experiments/exp2_lowrank_probe/lowrank_probe.py \
  --feat_dir "$SRC/features" \
  --out_dir "$OUT" \
  --epochs 80

# Once probe is done the heavy intermediates can go.
rm -f "$SRC/features/svd_accum.pt" "$SRC/features/pooled_feats.pt"
echo "[cleanup] removed pooled_feats.pt and svd_accum.pt from $SRC/features"
echo "[done] $OUT"
