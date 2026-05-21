#!/usr/bin/env bash
#SBATCH --job-name=ijepa-exp1-vitb
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a/aryanj/logs/ijepa-exp1-vitb-%j.out
#SBATCH --error=/home/a/aryanj/logs/ijepa-exp1-vitb-%j.err

set -euo pipefail

PROJ=$HOME/projects/ijepa
CKPT=$PROJ/checkpoints/mae_pretrain_vit_base.pth
DATA=/home/h/haoyu/ImageNetV2-matched-frequency
RUN_ID=${SLURM_JOB_ID:-local}
OUT=$PROJ/outputs/exp1_svd_vitb/${RUN_ID}
FEAT=$OUT/features

mkdir -p "$OUT" "$FEAT" "$HOME/logs"
source $PROJ/ijepaenv/bin/activate
cd $PROJ

echo "[1/4] extract features (streaming covariance)"
python experiments/exp1_svd_vitb/extract_features.py \
  --ckpt "$CKPT" \
  --data_root "$DATA" \
  --out_dir "$FEAT" \
  --batch_size 128 --num_workers 8 --dtype bf16

echo "[2/4] SVD analysis"
python experiments/exp1_svd_vitb/svd_analysis.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT/svd"

# svd_accum.pt (~60MB) no longer needed once metrics are saved.
rm -f "$FEAT/svd_accum.pt"
echo "[cleanup] removed svd_accum.pt"

echo "[3/4] linear probe (C-sweep)"
python experiments/exp1_svd_vitb/linear_probe.py \
  --feat_dir "$FEAT" \
  --out_dir "$OUT/probe"

echo "[4/4] comparison table"
python experiments/exp1_svd_vitb/make_table.py \
  --svd_json "$OUT/svd/svd_metrics.json" \
  --probe_json "$OUT/probe/linear_probe.json" \
  --out_md "$OUT/comparison_table.md" \
  --out_csv "$OUT/comparison_table.csv"

echo "[done] $OUT"
