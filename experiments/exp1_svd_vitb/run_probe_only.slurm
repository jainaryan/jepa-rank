#!/usr/bin/env bash
#SBATCH --job-name=ijepa-vitb-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/a/aryanj/logs/ijepa-vitb-probe-%j.out
#SBATCH --error=/home/a/aryanj/logs/ijepa-vitb-probe-%j.err

set -euo pipefail
PROJ=$HOME/projects/ijepa

# Reuse the artifacts from the cancelled extract+SVD run (618432).
# Override SRC_RUN_ID via env var if you want to re-run probe against a different run.
SRC_RUN_ID="${SRC_RUN_ID:-618432}"
SRC="$PROJ/outputs/exp1_svd_vitb/${SRC_RUN_ID}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUT="$PROJ/outputs/exp1_svd_vitb/${RUN_ID}"
mkdir -p "$OUT" "$HOME/logs"

# Symlink features + svd into the new run dir so all artifacts colocate.
ln -sfn "$SRC/features" "$OUT/features"
ln -sfn "$SRC/svd" "$OUT/svd"

source "$PROJ/ijepaenv/bin/activate"
cd "$PROJ"

echo "[probe] C-sweep on $SRC/features/pooled_feats.pt"
python experiments/exp1_svd_vitb/linear_probe.py \
  --feat_dir "$SRC/features" \
  --out_dir "$OUT/probe" \
  --epochs 100

echo "[table]"
python experiments/exp1_svd_vitb/make_table.py \
  --svd_json "$SRC/svd/svd_metrics.json" \
  --probe_json "$OUT/probe/linear_probe.json" \
  --out_md "$OUT/comparison_table.md" \
  --out_csv "$OUT/comparison_table.csv"

echo "[done] $OUT"
