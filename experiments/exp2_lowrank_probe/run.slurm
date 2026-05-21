#!/usr/bin/env bash
#SBATCH --job-name=jepa-rank-exp2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/a/aryanj/logs/jepa-rank-exp2-%j.out
#SBATCH --error=/home/a/aryanj/logs/jepa-rank-exp2-%j.err

set -euo pipefail

PROJ=$HOME/projects/ijepa
CKPT=$PROJ/checkpoints/IN1K-vit.h.14-300e.pth.tar
RUN_ID=${SLURM_JOB_ID:-local}
OUT=$PROJ/outputs/exp2_lowrank_probe/${RUN_ID}
FEAT=$OUT/features

mkdir -p "$OUT" "$FEAT" "$HOME/logs"
source "$PROJ/ijepaenv/bin/activate"
cd "$PROJ"

echo "[1/2] extract I-JEPA ViT-H/14 features + streaming covariance"
python experiments/exp2_lowrank_probe/extract_with_accum.py \
  --ckpt "$CKPT" \
  --out_dir "$FEAT" \
  --img_size 224 --batch_size 64 --num_workers 8 --dtype bf16

echo "[2/2] low-rank probe sweep"
python experiments/exp2_lowrank_probe/lowrank_probe.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT" \
  --epochs 80

# Clean up large intermediates (we keep the JSON/PNG/labels.pt artifacts).
rm -f "$FEAT/svd_accum.pt" "$FEAT/pooled_feats.pt"
echo "[cleanup] removed pooled_feats.pt and svd_accum.pt"

echo "[done] $OUT"
