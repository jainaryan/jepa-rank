"""
Extract per-layer features from a pretrained I-JEPA encoder on ImageNetV2.

Outputs (in --out_dir):
  pooled_feats.pt   dict[layer_idx] -> tensor (N_imgs, D)        mean-pooled tokens
  token_feats.pt    dict[layer_idx] -> tensor (N_svd, T, D)      raw token grids (subset for SVD)
  labels.pt         tensor (N_imgs,)
  meta.json         model/data/run metadata

Conventions:
  * "layer_idx" = 1-indexed block id (1..depth). layer 0 = post-patch-embed pre-block.
  * Final entry layer_idx = depth+1 corresponds to features after the final LayerNorm
    (`self.norm`), matching what downstream code consumes.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset: ImageNetV2-matched-frequency layout is <root>/<class_id>/<hash>.jpeg
# class_id is 0..999 already aligned with the ImageNet-1K class index.
# ---------------------------------------------------------------------------
class ImageNetV2(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for cls_dir in sorted(self.root.iterdir(), key=lambda p: int(p.name)):
            if not cls_dir.is_dir():
                continue
            label = int(cls_dir.name)
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
def load_ijepa_encoder(ckpt_path, arch, patch_size, img_size, device):
    """Build encoder via upstream code and load `target_encoder` weights."""
    from src.models import vision_transformer as vit

    model = vit.__dict__[arch](patch_size=patch_size, img_size=[img_size])
    model = model.to(device)

    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("target_encoder", ckpt.get("encoder"))
    # strip "module." prefix from DDP
    state = {k.removeprefix("module."): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print("  missing[:5]:", msg.missing_keys[:5])
    if msg.unexpected_keys:
        print("  unexpected[:5]:", msg.unexpected_keys[:5])

    model.eval()
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/home/h/haoyu/ImageNetV2-matched-frequency")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", default="vit_huge", choices=["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge", "vit_giant"])
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--svd_subset", type=int, default=512,
                    help="number of images for which to also save raw token grids")
    ap.add_argument("--upstream_path", default=os.path.expanduser("~/projects/ijepa/upstream"))
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    sys.path.insert(0, args.upstream_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    # ImageNet normalization (I-JEPA's transforms.py uses these)
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    tfm = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    ds = ImageNetV2(args.data_root, transform=tfm)
    print(f"[data] {len(ds)} images from {args.data_root}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = load_ijepa_encoder(args.ckpt, args.arch, args.patch_size, args.img_size, device)

    depth = len(model.blocks)
    print(f"[model] arch={args.arch} depth={depth} embed_dim={model.embed_dim}")

    # ------------------------------------------------------------------
    # Forward hooks: capture output of every block. We also capture the
    # pre-blocks input and the post-norm output.
    # ------------------------------------------------------------------
    captured = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            captured[name] = out
        return hook

    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(make_hook(f"block_{i+1}")))
    if model.norm is not None:
        handles.append(model.norm.register_forward_hook(make_hook(f"block_{depth+1}_norm")))

    # ------------------------------------------------------------------
    # Allocate output buffers.
    # pooled: (N_imgs, D) per layer, fp32 on CPU.
    # tokens: (svd_subset, T, D) per layer, fp32 on CPU (only first svd_subset images).
    # ------------------------------------------------------------------
    N = len(ds)
    D = model.embed_dim
    n_layers = depth + 1  # blocks 1..depth + final-norm
    T = (args.img_size // args.patch_size) ** 2

    pooled = {l: torch.empty(N, D, dtype=torch.float32) for l in range(1, n_layers + 1)}
    token_subset_n = min(args.svd_subset, N)
    tokens = {l: torch.empty(token_subset_n, T, D, dtype=torch.float32) for l in range(1, n_layers + 1)}
    labels = torch.empty(N, dtype=torch.long)

    cursor = 0
    for imgs, lbls in tqdm(loader, desc="extract"):
        imgs = imgs.to(device, non_blocking=True)
        bsz = imgs.size(0)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            _ = model(imgs)

        # blocks 1..depth, plus the final-norm layer at depth+1
        for l in range(1, depth + 1):
            feats = captured[f"block_{l}"].float()  # (B, T, D)
            pooled[l][cursor:cursor + bsz] = feats.mean(dim=1).cpu()
            if cursor < token_subset_n:
                end = min(token_subset_n, cursor + bsz)
                tokens[l][cursor:end] = feats[: end - cursor].cpu()

        # final layer-normed
        feats = captured[f"block_{depth+1}_norm"].float()
        pooled[n_layers][cursor:cursor + bsz] = feats.mean(dim=1).cpu()
        if cursor < token_subset_n:
            end = min(token_subset_n, cursor + bsz)
            tokens[n_layers][cursor:end] = feats[: end - cursor].cpu()

        labels[cursor:cursor + bsz] = lbls
        cursor += bsz

    for h in handles:
        h.remove()

    torch.save(pooled, out_dir / "pooled_feats.pt")
    torch.save(tokens, out_dir / "token_feats.pt")
    torch.save(labels, out_dir / "labels.pt")
    meta = {
        "ckpt": args.ckpt,
        "arch": args.arch,
        "patch_size": args.patch_size,
        "img_size": args.img_size,
        "depth": depth,
        "n_layers_saved": n_layers,
        "embed_dim": D,
        "tokens_per_image": T,
        "n_images": N,
        "svd_subset": token_subset_n,
        "data_root": args.data_root,
        "dtype": args.dtype,
        "note": (
            "layer indices 1..depth correspond to outputs of block_i; "
            "layer index depth+1 corresponds to features after final LayerNorm."
        ),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
