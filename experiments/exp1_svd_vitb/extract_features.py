"""
Extract per-layer features from MAE-B/16 on ImageNetV2 (substitute for I-JEPA-B/16).

Differences from the ViT-H/14 run:
  * Uses MAE-pretrained ViT-B/16 (Meta has not released I-JEPA-B/16).
  * Streams the per-layer covariance X^T X instead of saving raw token tensors.
    For each batch and each layer we accumulate:
        XtX_l += X^T X        (D, D)
        sum_l += X.sum(0)     (D,)
        n_l   += X.shape[0]
    After the full pass, centered covariance:
        C = XtX - n * mu mu^T       (mu = sum / n)
    Eigenvalues of C give sigma_i^2 in the centered SVD. No token tensors stored.
  * CLS token excluded from both pooled features and SVD accumulators (matches
    I-JEPA's no-CLS scheme).

Outputs in --out_dir:
  pooled_feats.pt   dict[layer] -> (N_imgs, D)  fp32  mean-pool over patch tokens
  svd_accum.pt      dict[layer] -> {"XtX": (D,D) fp64, "sum": (D,) fp64, "n": int}
  labels.pt         (N_imgs,) long
  meta.json
"""

import argparse
import json
import os
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ---- ImageNetV2 dataset ----
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
        return self.transform(Image.open(path).convert("RGB")), label


def load_mae_vit_b16(ckpt_path, device):
    """
    Build a timm ViT-B/16 and load MAE pretrain weights.

    The MAE checkpoint stores the pretrain model under the key "model" and
    includes decoder weights we don't care about. We load with strict=False so
    the decoder keys (mask_token, decoder_*) are ignored.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",      # return tokens, not pooled
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    # MAE keys are aligned with timm ViT for the encoder side.
    msg = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print("  missing[:5]:", msg.missing_keys[:5])
    if msg.unexpected_keys:
        print("  unexpected[:5]:", msg.unexpected_keys[:5])
    return model.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/home/h/haoyu/ImageNetV2-matched-frequency")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    # Standard ImageNet normalization at 224.
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    ds = ImageNetV2(args.data_root, transform=tfm)
    print(f"[data] {len(ds)} images")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = load_mae_vit_b16(args.ckpt, device)
    depth = len(model.blocks)
    D = model.embed_dim
    print(f"[model] timm vit_base_patch16_224 (MAE) depth={depth} embed_dim={D}")

    # ---- hooks ----
    captured = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            captured[name] = out
        return hook

    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(make_hook(f"block_{i+1}")))
    handles.append(model.norm.register_forward_hook(make_hook(f"block_{depth+1}_norm")))

    n_layers = depth + 1
    N = len(ds)
    layer_keys = [f"block_{i}" for i in range(1, depth + 1)] + [f"block_{depth+1}_norm"]

    # ---- output buffers ----
    pooled = {l: torch.empty(N, D, dtype=torch.float32) for l in range(1, n_layers + 1)}
    accum = {
        l: {
            "XtX": torch.zeros(D, D, dtype=torch.float64),
            "sum": torch.zeros(D, dtype=torch.float64),
            "n": 0,
        } for l in range(1, n_layers + 1)
    }
    labels = torch.empty(N, dtype=torch.long)

    cursor = 0
    for imgs, lbls in tqdm(loader, desc="extract"):
        imgs = imgs.to(device, non_blocking=True)
        bsz = imgs.size(0)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            _ = model(imgs)

        for l_idx, key in enumerate(layer_keys, start=1):
            feats = captured[key]                                  # (B, 1+T, D)
            patch = feats[:, 1:, :].float()                         # drop CLS
            # ---- pooled features for probe ----
            pooled[l_idx][cursor:cursor + bsz] = patch.mean(dim=1).cpu()
            # ---- streaming covariance accumulators (on GPU then transfer) ----
            X = patch.reshape(-1, D).double()                       # (B*T, D)
            accum[l_idx]["XtX"] += (X.T @ X).cpu()
            accum[l_idx]["sum"] += X.sum(dim=0).cpu()
            accum[l_idx]["n"] += X.shape[0]

        labels[cursor:cursor + bsz] = lbls
        cursor += bsz

    for h in handles:
        h.remove()

    torch.save(pooled, out_dir / "pooled_feats.pt")
    torch.save(accum, out_dir / "svd_accum.pt")
    torch.save(labels, out_dir / "labels.pt")

    meta = {
        "model": "timm/vit_base_patch16_224 (MAE pretrain)",
        "ckpt": args.ckpt,
        "ckpt_size_bytes": os.path.getsize(args.ckpt),
        "depth": depth,
        "n_layers_saved": n_layers,
        "embed_dim": D,
        "patch_size": 16,
        "img_size": 224,
        "tokens_per_image": 196,
        "cls_excluded": True,
        "n_images": N,
        "data_root": args.data_root,
        "dtype": args.dtype,
        "note": (
            "SVD via streaming covariance: XtX, sum, n accumulators saved per layer. "
            "Centered Gram = XtX - n * mu mu^T; sigma_i^2 = eigvals(centered Gram)."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
