"""
SVD / effective-rank analysis of per-layer token features.

Inputs:  <feat_dir>/token_feats.pt  (dict[layer] -> (N, T, D))
Outputs: <out_dir>/svd_metrics.json
         <out_dir>/svd_spectrum.png            log-log singular value spectra
         <out_dir>/effective_rank_by_layer.png effective-rank line chart
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def effective_rank_metrics(M):
    """
    M: (N, D) float matrix on CPU.
    Returns dict of:
      sing_vals      : sorted singular values (descending), numpy
      stable_rank    : ||M||_F^2 / sigma_max^2
      participation  : (sum sigma)^2 / sum(sigma^2)
      rank_at_95     : smallest k s.t. cumulative energy >= 95%
      rank_at_99     : same for 99%
      shannon_rank   : exp(entropy of normalized sigma^2)
    """
    # use torch.linalg.svdvals (only values, faster)
    s = torch.linalg.svdvals(M).cpu().numpy().astype(np.float64)
    s = np.sort(s)[::-1]
    energy = s ** 2
    total = energy.sum()
    cum = np.cumsum(energy) / total

    stable_rank = float(total / (s[0] ** 2))
    participation = float((s.sum() ** 2) / energy.sum())  # PR on sigma (not energy)
    rank_at_95 = int(np.searchsorted(cum, 0.95) + 1)
    rank_at_99 = int(np.searchsorted(cum, 0.99) + 1)

    p = energy / total
    p_safe = np.clip(p, 1e-20, None)
    shannon = float(np.exp(-(p_safe * np.log(p_safe)).sum()))

    return {
        "sing_vals": s,
        "stable_rank": stable_rank,
        "participation": participation,
        "rank_at_95": rank_at_95,
        "rank_at_99": rank_at_99,
        "shannon_rank": shannon,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=200_000,
                    help="cap tokens per layer for SVD (subsample if more)")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = torch.load(feat_dir / "token_feats.pt", map_location="cpu", weights_only=False)
    meta = json.loads((feat_dir / "meta.json").read_text())
    layers = sorted(tokens.keys())

    metrics = {}
    spectra = {}
    rng = np.random.default_rng(0)

    for l in layers:
        F = tokens[l]                       # (N, T, D)
        N, T, D = F.shape
        M = F.reshape(N * T, D)             # (N*T, D)
        if M.size(0) > args.max_tokens:
            idx = rng.choice(M.size(0), args.max_tokens, replace=False)
            M = M[torch.from_numpy(idx)]
        # center: subtract mean — SVD on centered data gives PCA-style spectrum
        M = M - M.mean(dim=0, keepdim=True)
        res = effective_rank_metrics(M)
        spectra[l] = res.pop("sing_vals")
        metrics[l] = res
        print(f"layer {l:>2}: stable={res['stable_rank']:.1f}  "
              f"rank95={res['rank_at_95']}  rank99={res['rank_at_99']}  "
              f"shannon={res['shannon_rank']:.1f}  D={D}")

    # ---- plot spectra ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, l in enumerate(layers):
        s = spectra[l]
        ax.plot(np.arange(1, len(s) + 1), s / s[0],
                color=cmap(i / max(1, len(layers) - 1)),
                label=f"L{l}" if (l == 1 or l == layers[-1] or l % max(1, len(layers) // 6) == 0) else None,
                linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("$\\sigma_i / \\sigma_1$")
    ax.set_title(f"per-layer SVD spectra — {meta['arch']} ({meta['depth']} blocks, d={meta['embed_dim']})")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "svd_spectrum.png", dpi=150)
    plt.close(fig)

    # ---- plot effective rank vs layer ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color in [("stable_rank", "C0"), ("rank_at_95", "C1"),
                       ("rank_at_99", "C2"), ("shannon_rank", "C3")]:
        ax.plot(layers, [metrics[l][key] for l in layers], "o-", color=color, label=key)
    ax.axhline(meta["embed_dim"], color="k", linestyle="--", alpha=0.5, label="d (full)")
    ax.set_xlabel("layer index")
    ax.set_ylabel("rank")
    ax.set_title("effective rank across depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_rank_by_layer.png", dpi=150)
    plt.close(fig)

    # ---- save json ----
    out = {"layers": layers, "embed_dim": meta["embed_dim"],
           "metrics": {str(l): metrics[l] for l in layers}}
    with open(out_dir / "svd_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
