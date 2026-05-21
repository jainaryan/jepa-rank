#!/usr/bin/env bash
# MAE-B/16 pretrained on IN-1K (400 epochs, pixel reconstruction).
# Used as a ViT-B/16 SSL stand-in for I-JEPA (which has no public B/16 release).
set -euo pipefail
CKPT_DIR="${CKPT_DIR:-$HOME/projects/ijepa/checkpoints}"
mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"
URL="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
FNAME="mae_pretrain_vit_base.pth"
if [[ -f "$FNAME" ]]; then
  echo "[skip] $FNAME ($(du -h "$FNAME" | cut -f1))"
else
  echo "[download] $URL"
  curl -L --fail -o "$FNAME.partial" "$URL"
  mv "$FNAME.partial" "$FNAME"
  echo "[done] $(du -h "$FNAME" | cut -f1)"
fi
