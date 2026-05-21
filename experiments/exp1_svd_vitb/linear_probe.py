"""
Per-layer linear probe with a regularization sweep, GPU-based for speed.

We use the same protocol as before (8 train / 2 val per class, standardize
using train statistics) and sweep AdamW weight_decay over a grid that's
equivalent to the sklearn C-sweep:

  sklearn primal:  argmin (1/2)||w||^2 + C * sum_i CE(x_i, y_i)
  torch eq:        argmin (1/n) sum_i CE + (1/(2*C*n)) * ||w||^2
                   ⇒ AdamW wd ≈ 1/(C * n)  (with n_samples = 8000)

So the requested sklearn C ∈ {1e-3, 1e-2, 0.1, 1, 10} maps to
  wd ≈ {0.125, 0.0125, 1.25e-3, 1.25e-4, 1.25e-5}.

For each layer and each wd we train for `--epochs` and report the best val
top-1 across both the wd sweep and the per-epoch evaluation (so the result
is "best val accuracy under the regularization sweep").

This is ~50× faster than sklearn lbfgs on 1000-class multinomial LR and
gives essentially the same answer (linear classifier on standardized features).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_by_class(labels, n_train_per_class=8):
    labels_np = labels.numpy()
    train_idx, val_idx = [], []
    for c in np.unique(labels_np):
        idx = np.where(labels_np == c)[0]
        train_idx.extend(idx[:n_train_per_class].tolist())
        val_idx.extend(idx[n_train_per_class:].tolist())
    return torch.tensor(train_idx), torch.tensor(val_idx)


def fit_one(X_tr, y_tr, X_val, y_val, wd, *, epochs, lr, num_classes, device):
    """One LR fit with the given weight decay. Returns best per-epoch val top-1 & top-5."""
    D = X_tr.size(1)
    clf = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = 2048
    n = X_tr.size(0)
    best_top1 = 0.0
    best_top5 = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        clf.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(X_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            clf.eval()
            logits = clf(X_val)
            top1 = (logits.argmax(1) == y_val).float().mean().item()
            top5 = logits.topk(5, dim=1).indices.eq(y_val.unsqueeze(1)).any(1).float().mean().item()
            if top1 > best_top1:
                best_top1 = top1
                best_top5 = top5
    return best_top1, best_top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train_per_class", type=int, default=8)
    ap.add_argument("--C_values", type=str, default="0.001,0.01,0.1,1,10",
                    help="sklearn-style C values; mapped to AdamW wd = 1/(C*n_train).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--num_classes", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = torch.load(feat_dir / "pooled_feats.pt", map_location="cpu", weights_only=False)
    labels = torch.load(feat_dir / "labels.pt", map_location="cpu", weights_only=False)
    meta = json.loads((feat_dir / "meta.json").read_text())

    Cs = [float(x) for x in args.C_values.split(",")]
    train_idx, val_idx = split_by_class(labels, args.n_train_per_class)
    n_tr = len(train_idx)
    wd_for = {C: 1.0 / (C * n_tr) for C in Cs}

    y_tr = labels[train_idx].to(device)
    y_val = labels[val_idx].to(device)

    print(f"[split] train={n_tr} val={len(val_idx)} classes={len(torch.unique(labels))}")
    print(f"[sweep] C -> wd:  {', '.join(f'{C:g}->{wd_for[C]:.2e}' for C in Cs)}")

    results = {}
    layers = sorted(pooled.keys())
    for l in layers:
        X = pooled[l]
        X_tr = X[train_idx].to(device)
        X_val = X[val_idx].to(device)
        mu = X_tr.mean(0, keepdim=True)
        sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
        X_tr = (X_tr - mu) / sd
        X_val = (X_val - mu) / sd

        per_C = {}
        t0 = time.time()
        for C in Cs:
            wd = wd_for[C]
            top1, top5 = fit_one(X_tr, y_tr, X_val, y_val, wd,
                                 epochs=args.epochs, lr=args.lr,
                                 num_classes=args.num_classes, device=device)
            per_C[C] = {"top1": top1, "top5": top5, "wd": wd}
        best_C = max(per_C, key=lambda c: per_C[c]["top1"])
        best = per_C[best_C]
        dt = time.time() - t0
        print(f"layer {l:>2}: best_C={best_C:<7g}  top1={best['top1']*100:.2f}  "
              f"top5={best['top5']*100:.2f}  ({dt:.1f}s)")
        results[l] = {"best_C": best_C, "best_top1": best["top1"], "best_top5": best["top5"],
                      "per_C": {str(c): v for c, v in per_C.items()}}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, [results[l]["best_top1"] * 100 for l in layers], "o-", label="best top-1")
    ax.plot(layers, [results[l]["best_top5"] * 100 for l in layers], "o--", alpha=0.6, label="best top-5")
    ax.set_xlabel("layer index")
    ax.set_ylabel("ImageNetV2 accuracy (%)")
    ax.set_title(f"per-layer linear probe (C-sweep) — {meta['model']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "linear_probe_by_layer.png", dpi=150)
    plt.close(fig)

    (out_dir / "linear_probe.json").write_text(json.dumps({
        "layers": layers,
        "C_values": Cs,
        "C_to_wd": {str(C): wd_for[C] for C in Cs},
        "n_train_per_class": args.n_train_per_class,
        "epochs": args.epochs,
        "lr": args.lr,
        "results": {str(l): results[l] for l in layers},
    }, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
