#!/usr/bin/env bash
# Pull pretrained I-JEPA ViT-H/14 (300ep IN-1K) checkpoint.
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-$HOME/projects/ijepa/checkpoints}"
mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"

URL="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
FNAME="IN1K-vit.h.14-300e.pth.tar"

if [[ -f "$FNAME" ]]; then
  echo "[skip] $FNAME already present ($(du -h "$FNAME" | cut -f1))"
else
  echo "[download] $URL"
  curl -L --fail -o "$FNAME.partial" "$URL"
  mv "$FNAME.partial" "$FNAME"
  echo "[done] $(du -h "$FNAME" | cut -f1)"
fi
