#!/usr/bin/env bash
# Rsync this project to the NUS cluster. GitHub SSH is blocked on the cluster,
# so rsync is how we ship code.
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="nus-student-cluster:~/projects/ijepa/"

rsync -az --delete \
  --exclude '.git/' \
  --exclude 'upstream/' \
  --exclude 'checkpoints/' \
  --exclude 'outputs/' \
  --exclude 'features_cache/' \
  --exclude 'ijepaenv/' \
  --exclude '.tmp/' \
  --exclude '.claude/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  -e "ssh -o LogLevel=QUIET" \
  "$LOCAL_ROOT/" "$REMOTE"

echo "[sync] -> $REMOTE"
