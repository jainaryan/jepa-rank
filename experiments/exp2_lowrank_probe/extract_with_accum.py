"""
Re-extract per-layer features from I-JEPA ViT-H/14 with the streaming
covariance accumulator (so we can recover the eigenvectors used by the
low-rank probe). Identical methodology to the MAE/DINOv2 runs, but for the
I-JEPA encoder (no CLS token).

Saves (in --out_dir):
  pooled_feats.pt    dict[layer] -> (N_imgs, D)  fp32  mean-pooled tokens
  svd_accum.pt       dict[layer] -> {"XtX": (D,D) fp64, "sum": (D,) fp64, "n": int}
  labels.pt          (N_imgs,) long
  meta.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


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


def load_ijepa_huge(ckpt_path, img_size, device, upstream_path):
    sys.path.insert(0, upstream_path)
    from src.models import vision_transformer as vit

    model = vit.vit_huge(patch_size=14, img_size=[img_size])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("target_encoder", ckpt.get("encoder"))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/home/h/haoyu/ImageNetV2-matched-frequency")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--upstream_path", default=os.path.expanduser("~/projects/ijepa/upstream"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    tfm = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ds = ImageNetV2(args.data_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"[data] {len(ds)} images")

    model = load_ijepa_huge(args.ckpt, args.img_size, device, args.upstream_path)
    depth = len(model.blocks)
    D = model.embed_dim
    tokens_per_img = (args.img_size // 14) ** 2
    print(f"[model] I-JEPA vit_huge/14 depth={depth} embed_dim={D} tokens/img={tokens_per_img}")

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

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype != torch.float32)):
            _ = model(imgs)

        for l_idx, key in enumerate(layer_keys, start=1):
            feats = captured[key].float()           # (B, T, D)  — no CLS in I-JEPA
            pooled[l_idx][cursor:cursor + bsz] = feats.mean(dim=1).cpu()
            X = feats.reshape(-1, D).double()
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
    (out_dir / "meta.json").write_text(json.dumps({
        "model": "I-JEPA vit_huge_patch14 (IN1K 300e)",
        "ckpt": args.ckpt,
        "depth": depth,
        "n_layers_saved": n_layers,
        "embed_dim": D,
        "patch_size": 14,
        "img_size": args.img_size,
        "tokens_per_image": tokens_per_img,
        "cls_excluded": False,
        "n_images": N,
        "data_root": args.data_root,
        "dtype": args.dtype,
    }, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
