"""
Extract per-layer features from DINOv2 ViT-B/14 on ImageNetV2.

Identical methodology to the MAE-B/16 run (streaming covariance, mean-pool over
patch tokens with CLS excluded, fp32 pooled features). The only differences:
  * timm model id: vit_base_patch14_dinov2.lvd142m  (no register tokens)
  * patch_size = 14 → 256 patch tokens at 224 input
  * timm pretrained=True downloads weights from Hugging Face on first run
"""

import argparse
import json
from pathlib import Path

import timm
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


def load_dinov2_vit_b14(img_size, device):
    """
    Load DINOv2 ViT-B/14 via timm with pretrained=True. timm fetches from
    Hugging Face on first run and caches to ~/.cache/huggingface/hub/.
    The 'lvd142m' tag = pretrained on the LVD-142M dataset (DINOv2 paper).
    """
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        img_size=img_size,        # interpolates pos-embed from native 518 → img_size
        num_classes=0,
        global_pool="",           # return raw token output
    )
    return model.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/h/haoyu/ImageNetV2-matched-frequency")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    ds = ImageNetV2(args.data_root, transform=tfm)
    print(f"[data] {len(ds)} images")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = load_dinov2_vit_b14(args.img_size, device)
    depth = len(model.blocks)
    D = model.embed_dim
    tokens_per_img = (args.img_size // 14) ** 2
    print(f"[model] DINOv2 vit_base_patch14 (lvd142m) depth={depth} embed_dim={D} "
          f"tokens/img={tokens_per_img}")

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
            pooled[l_idx][cursor:cursor + bsz] = patch.mean(dim=1).cpu()
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
        "model": "timm/vit_base_patch14_dinov2.lvd142m (DINOv2)",
        "depth": depth,
        "n_layers_saved": n_layers,
        "embed_dim": D,
        "patch_size": 14,
        "img_size": args.img_size,
        "tokens_per_image": tokens_per_img,
        "cls_excluded": True,
        "n_images": N,
        "data_root": args.data_root,
        "dtype": args.dtype,
        "register_tokens": 0,
        "note": (
            "SVD via streaming covariance: XtX, sum, n accumulators saved per layer. "
            "Centered Gram = XtX - n*mu*mu^T; sigma_i^2 = eigvals(centered Gram). "
            "Note pos-embed interpolated from native 518 to img_size."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
