"""
Low-rank probe sweep.

Question this answers, layer by layer:
  "If layer L only had k principal directions of its token features, how much
   does the linear probe lose vs the full-d probe?"

Method, per layer L:
  1. Eigendecompose the centered token-Gram matrix (from svd_accum) → eigenvectors
     V_L (D, D), columns sorted by descending singular value.
  2. For each k in the sweep, project the *pooled* feature vectors onto the
     top-k subspace:  Z_k = pooled[L] @ V_L[:, :k]   (shape (N, k)).
     This equals "mean-pool of token features projected onto top-k", because
     mean-pool commutes with the linear projection.
  3. Train a 1000-way linear classifier on Z_k (8 train / 2 val per class,
     standardize on train stats, sweep C ∈ {1e-3,1e-2,0.1,1,10}, report best
     val top-1).

For each layer we also evaluate two distinguished k values from the original
ViT-H/14 SVD run:
  * k = rank@95(L) — the question's headline
  * k = rank@99(L) — slightly more dimensions, expected to be near-lossless
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


# ----------------------------------------------------------------------
def split_by_class(labels, n_train_per_class=8):
    labels_np = labels.numpy()
    train_idx, val_idx = [], []
    for c in np.unique(labels_np):
        idx = np.where(labels_np == c)[0]
        train_idx.extend(idx[:n_train_per_class].tolist())
        val_idx.extend(idx[n_train_per_class:].tolist())
    return torch.tensor(train_idx), torch.tensor(val_idx)


def fit_one(X_tr, y_tr, X_val, y_val, wd, *, epochs, lr, num_classes, device):
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


def best_over_C(X_tr, y_tr, X_val, y_val, Cs, n_tr, *, epochs, lr, num_classes, device):
    best = {"top1": 0.0, "top5": 0.0, "C": None}
    for C in Cs:
        wd = 1.0 / (C * n_tr)
        top1, top5 = fit_one(X_tr, y_tr, X_val, y_val, wd,
                             epochs=epochs, lr=lr,
                             num_classes=num_classes, device=device)
        if top1 > best["top1"]:
            best = {"top1": top1, "top5": top5, "C": C}
    return best


# ----------------------------------------------------------------------
def eigvecs_from_accum(XtX, s, n):
    """Eigenvectors of the centered Gram, sorted by descending eigenvalue."""
    mu = s / n
    C = XtX - n * torch.outer(mu, mu)
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = torch.linalg.eigh(C)
    # eigh returns ascending; flip to descending
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = torch.clamp(eigvals, min=0.0)
    return eigvecs, eigvals


def rank_at_energy(eigvals, frac):
    e = eigvals.numpy()
    cum = np.cumsum(e) / e.sum()
    return int(np.searchsorted(cum, frac) + 1)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", type=str, default="",
                    help="Comma-separated list of layer indices (1-based). "
                         "Empty = every-other layer + the post-norm.")
    ap.add_argument("--k_values", type=str,
                    default="4,8,16,32,64,128,256,384,512,768,1024,1280")
    ap.add_argument("--C_values", type=str, default="0.001,0.01,0.1,1,10")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--num_classes", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = torch.load(feat_dir / "pooled_feats.pt", map_location="cpu", weights_only=False)
    accum = torch.load(feat_dir / "svd_accum.pt", map_location="cpu", weights_only=False)
    labels = torch.load(feat_dir / "labels.pt", map_location="cpu", weights_only=False)
    meta = json.loads((feat_dir / "meta.json").read_text())
    D = meta["embed_dim"]

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        depth = meta["depth"]
        layers = list(range(1, depth + 1, 2)) + [depth, depth + 1]
        layers = sorted(set(layers))
    Cs = [float(x) for x in args.C_values.split(",")]
    ks_base = [int(x) for x in args.k_values.split(",") if int(x) <= D]

    train_idx, val_idx = split_by_class(labels, n_train_per_class=8)
    n_tr = len(train_idx)
    y_tr_all = labels[train_idx].to(device)
    y_val_all = labels[val_idx].to(device)

    print(f"[setup] layers={layers}")
    print(f"[setup] k base grid={ks_base} (+ rank@95 and rank@99 per layer)")
    print(f"[setup] C grid={Cs}, epochs={args.epochs}")
    print(f"[split] train={n_tr} val={len(val_idx)}")

    all_results = {}
    for l in layers:
        a = accum[l]
        eigvecs, eigvals = eigvecs_from_accum(a["XtX"], a["sum"], a["n"])
        r95 = rank_at_energy(eigvals, 0.95)
        r99 = rank_at_energy(eigvals, 0.99)
        ks = sorted(set(ks_base + [r95, r99]))
        ks = [k for k in ks if 1 <= k <= D]

        eigvecs_dev = eigvecs.float().to(device)         # (D, D)
        X = pooled[l].to(device)                          # (N, D)
        # Project ALL pooled vectors onto eigenbasis once; then slice per k.
        X_eig_all = X @ eigvecs_dev                       # (N, D)

        results_for_layer = {}
        t0 = time.time()
        for k in ks:
            Z = X_eig_all[:, :k]                          # (N, k)
            Z_tr = Z[train_idx]
            Z_val = Z[val_idx]
            mu = Z_tr.mean(0, keepdim=True)
            sd = Z_tr.std(0, keepdim=True).clamp_min(1e-6)
            Z_tr = (Z_tr - mu) / sd
            Z_val = (Z_val - mu) / sd
            best = best_over_C(Z_tr, y_tr_all, Z_val, y_val_all, Cs, n_tr,
                               epochs=args.epochs, lr=args.lr,
                               num_classes=args.num_classes, device=device)
            results_for_layer[k] = best
        dt = time.time() - t0
        full_top1 = results_for_layer[D]["top1"] if D in results_for_layer else results_for_layer[max(ks)]["top1"]
        r95_top1 = results_for_layer[r95]["top1"]
        r99_top1 = results_for_layer[r99]["top1"]
        print(f"L{l:>2}: r95={r95:<4} r99={r99:<4} | "
              f"k=4: {results_for_layer[4]['top1']*100:5.2f}  "
              f"k=64: {results_for_layer[64]['top1']*100:5.2f}  "
              f"k=r95: {r95_top1*100:5.2f}  "
              f"k=r99: {r99_top1*100:5.2f}  "
              f"k={D}: {full_top1*100:5.2f}  "
              f"r95/full = {r95_top1/full_top1*100:5.1f}%  ({dt:.1f}s)")
        all_results[l] = {
            "ks": ks,
            "rank_at_95": r95,
            "rank_at_99": r99,
            "per_k": {str(k): results_for_layer[k] for k in ks},
            "full_d_top1": full_top1,
            "r95_top1": r95_top1,
            "r99_top1": r99_top1,
        }

    # ------- save -------
    (out_dir / "lowrank_probe.json").write_text(json.dumps({
        "layers": layers,
        "k_base": ks_base,
        "C_values": Cs,
        "epochs": args.epochs,
        "embed_dim": D,
        "results": {str(l): all_results[l] for l in layers},
    }, indent=2))

    # ------- main plot: top-1 vs k, one line per layer -------
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    for i, l in enumerate(layers):
        r = all_results[l]
        ks = r["ks"]
        ys = [r["per_k"][str(k)]["top1"] * 100 for k in ks]
        color = cmap(i / max(1, len(layers) - 1))
        ax.plot(ks, ys, "-o", color=color, label=f"L{l}", markersize=3, linewidth=1)
        # Mark rank@95
        ax.scatter([r["rank_at_95"]], [r["r95_top1"] * 100],
                   marker="*", s=70, edgecolor="k", color=color, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("k (projected dimension)")
    ax.set_ylabel("ImageNetV2 top-1 (%)")
    ax.set_title("Low-rank probe — I-JEPA ViT-H/14  (★ = k = rank@95 of that layer)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "lowrank_probe_vs_k.png", dpi=150)
    plt.close(fig)

    # ------- summary plot: full-d vs k=rank95(L) vs k=rank99(L), x = layer -------
    fig, ax = plt.subplots(figsize=(9, 5))
    full = [all_results[l]["full_d_top1"] * 100 for l in layers]
    r95s = [all_results[l]["r95_top1"] * 100 for l in layers]
    r99s = [all_results[l]["r99_top1"] * 100 for l in layers]
    ax.plot(layers, full, "o-", color="C0", label="full d (=1280)")
    ax.plot(layers, r99s, "o-", color="C2", label="k = rank@99(L)")
    ax.plot(layers, r95s, "o-", color="C1", label="k = rank@95(L)")
    ax.set_xlabel("layer")
    ax.set_ylabel("ImageNetV2 top-1 (%)")
    ax.set_title("Probe accuracy at low rank vs full d — I-JEPA ViT-H/14")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lowrank_probe_by_layer.png", dpi=150)
    plt.close(fig)

    # ------- table -------
    md = ["| layer | rank@95 | rank@99 | top-1 @ rank95 | top-1 @ rank99 | top-1 @ full d | r95/full |",
          "|------:|--------:|--------:|---------------:|---------------:|---------------:|---------:|"]
    for l in layers:
        r = all_results[l]
        f = r["full_d_top1"] * 100
        a = r["r95_top1"] * 100
        b = r["r99_top1"] * 100
        ratio = (a / f * 100) if f > 0 else float("nan")
        md.append(f"| {l} | {r['rank_at_95']} | {r['rank_at_99']} | "
                  f"{a:.2f} | {b:.2f} | {f:.2f} | {ratio:.1f}% |")
    (out_dir / "lowrank_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
