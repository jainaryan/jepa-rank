#!/usr/bin/env bash
#SBATCH --job-name=ijepa-exp1
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/aryanj/logs/ijepa-exp1-%j.out
#SBATCH --error=/home/a/aryanj/logs/ijepa-exp1-%j.err

set -euo pipefail

PROJ=$HOME/projects/ijepa
CKPT=$PROJ/checkpoints/IN1K-vit.h.14-300e.pth.tar
DATA=/home/h/haoyu/ImageNetV2-matched-frequency
RUN_ID=${SLURM_JOB_ID:-local}
OUT=$PROJ/outputs/exp1_svd/${RUN_ID}
FEAT=$OUT/features

mkdir -p "$OUT" "$FEAT" "$HOME/logs"

source $PROJ/ijepaenv/bin/activate

cd $PROJ

echo "[1/3] extract features"
python experiments/exp1_svd/extract_features.py \
  --ckpt "$CKPT" \
  --data_root "$DATA" \
  --out_dir "$FEAT" \
  --arch vit_huge --patch_size 14 --img_size 224 \
  --batch_size 64 --num_workers 8 \
  --svd_subset 512 --dtype bf16

echo "[2/3] SVD analysis"
python experiments/exp1_svd/svd_analysis.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT/svd"

# SVD metrics saved; token_feats.pt (~20G for ViT-H/14) no longer needed.
rm -f "$FEAT/token_feats.pt"
echo "[cleanup] removed token_feats.pt"

echo "[3/3] linear probe"
python experiments/exp1_svd/linear_probe.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT/probe" \
  --epochs 80

# Probe done; pooled_feats.pt (~1.6G for ViT-H/14) no longer needed either.
rm -f "$FEAT/pooled_feats.pt"
echo "[cleanup] removed pooled_feats.pt"

echo "[done] outputs at $OUT (kept: labels.pt, meta.json, svd/, probe/)"
