#!/usr/bin/env bash
# Optional: pre-download DINOv2 ViT-B/14 weights into the local HF cache so
# the SLURM job doesn't need internet on its first run. Safe to skip — the job
# will fall back to downloading on its own.
set -euo pipefail
PROJ=${PROJ:-$HOME/projects/ijepa}
source "$PROJ/ijepaenv/bin/activate"
python - <<'PY'
import timm
m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0)
print("[ok] DINOv2-B/14 weights cached. embed_dim =", m.embed_dim, "depth =", len(m.blocks))
PY
