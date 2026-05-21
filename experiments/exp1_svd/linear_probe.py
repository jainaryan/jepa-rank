"""
Per-layer linear probe on mean-pooled features.

Train a single linear classifier (1000-way) per layer on ImageNetV2:
  * 8 imgs/class train  (8000)
  * 2 imgs/class val    (2000)

Uses an in-GPU multinomial logistic regression trained with AdamW (fast, ~seconds/layer).

Inputs:  <feat_dir>/pooled_feats.pt, labels.pt, meta.json
Outputs: <out_dir>/linear_probe.json + linear_probe_by_layer.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def split_by_class(labels, n_train_per_class=8):
    """Deterministic 8/2 split per class — first n in dataset order are train."""
    labels_np = labels.numpy()
    train_idx, val_idx = [], []
    for c in np.unique(labels_np):
        idx = np.where(labels_np == c)[0]
        train_idx.extend(idx[:n_train_per_class].tolist())
        val_idx.extend(idx[n_train_per_class:].tolist())
    return torch.tensor(train_idx), torch.tensor(val_idx)


def fit_linear(X_tr, y_tr, X_val, y_val, num_classes, epochs=80, lr=1e-2, wd=1e-3, device="cuda"):
    """Train a single nn.Linear with AdamW + cosine schedule. Returns val top-1."""
    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # standardize using train statistics
    mu = X_tr.mean(0, keepdim=True)
    sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
    X_tr = (X_tr - mu) / sd
    X_val = (X_val - mu) / sd

    D = X_tr.size(1)
    clf = torch.nn.Linear(D, num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bs = 1024
    n = X_tr.size(0)
    best_top1 = 0.0
    for e in range(epochs):
        perm = torch.randperm(n, device=device)
        clf.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = clf(X_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        with torch.no_grad():
            clf.eval()
            logits = clf(X_val)
            top1 = (logits.argmax(1) == y_val).float().mean().item()
            top5 = logits.topk(5, dim=1).indices.eq(y_val.unsqueeze(1)).any(1).float().mean().item()
            best_top1 = max(best_top1, top1)

    return {"top1": top1, "top5": top5, "best_top1": best_top1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--n_train_per_class", type=int, default=8)
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = torch.load(feat_dir / "pooled_feats.pt", map_location="cpu", weights_only=False)
    labels = torch.load(feat_dir / "labels.pt", map_location="cpu", weights_only=False)
    meta = json.loads((feat_dir / "meta.json").read_text())

    train_idx, val_idx = split_by_class(labels, args.n_train_per_class)
    y_tr = labels[train_idx]
    y_val = labels[val_idx]
    print(f"[split] train={len(train_idx)} val={len(val_idx)} classes={len(torch.unique(labels))}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    layers = sorted(pooled.keys())
    for l in layers:
        X = pooled[l]
        X_tr = X[train_idx]
        X_val = X[val_idx]
        res = fit_linear(X_tr, y_tr, X_val, y_val,
                         num_classes=args.num_classes, epochs=args.epochs, device=device)
        print(f"layer {l:>2}: top1={res['top1']*100:.2f}  top5={res['top5']*100:.2f}  best_top1={res['best_top1']*100:.2f}")
        results[l] = res

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, [results[l]["best_top1"] * 100 for l in layers], "o-", label="best top-1")
    ax.plot(layers, [results[l]["top5"] * 100 for l in layers], "o--", alpha=0.6, label="final top-5")
    ax.set_xlabel("layer index")
    ax.set_ylabel("ImageNetV2 accuracy (%)")
    ax.set_title(f"per-layer linear probe — {meta['arch']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "linear_probe_by_layer.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "linear_probe.json", "w") as f:
        json.dump({"layers": layers,
                   "results": {str(l): results[l] for l in layers},
                   "n_train_per_class": args.n_train_per_class,
                   "epochs": args.epochs}, f, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
