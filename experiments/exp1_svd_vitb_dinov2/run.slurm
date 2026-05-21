#!/usr/bin/env bash
#SBATCH --job-name=jepa-rank-dinov2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/a/aryanj/logs/jepa-rank-dinov2-%j.out
#SBATCH --error=/home/a/aryanj/logs/jepa-rank-dinov2-%j.err

set -euo pipefail

PROJ=$HOME/projects/ijepa
RUN_ID=${SLURM_JOB_ID:-local}
OUT=$PROJ/outputs/exp1_svd_vitb_dinov2/${RUN_ID}
FEAT=$OUT/features

mkdir -p "$OUT" "$FEAT" "$HOME/logs"
source "$PROJ/ijepaenv/bin/activate"
cd "$PROJ"

# Allow timm to fetch the DINOv2 weights from HF on first run; subsequent
# runs use the cache under ~/.cache/huggingface.
export HF_HUB_ENABLE_HF_TRANSFER=1 || true

echo "[1/4] extract features (DINOv2-B/14, streaming covariance)"
python experiments/exp1_svd_vitb_dinov2/extract_features.py \
  --out_dir "$FEAT" \
  --img_size 224 --batch_size 128 --num_workers 8 --dtype bf16

echo "[2/4] SVD analysis (reusing MAE pipeline script — methodology is model-agnostic)"
python experiments/exp1_svd_vitb/svd_analysis.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT/svd"

rm -f "$FEAT/svd_accum.pt"
echo "[cleanup] removed svd_accum.pt"

echo "[3/4] linear probe (GPU C-sweep)"
python experiments/exp1_svd_vitb/linear_probe.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT/probe" \
  --epochs 100

echo "[4/4] comparison table"
python experiments/exp1_svd_vitb/make_table.py \
  --svd_json "$OUT/svd/svd_metrics.json" \
  --probe_json "$OUT/probe/linear_probe.json" \
  --out_md "$OUT/comparison_table.md" \
  --out_csv "$OUT/comparison_table.csv"

echo "[done] $OUT"
