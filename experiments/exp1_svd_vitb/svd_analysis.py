"""
SVD / effective-rank analysis from streamed covariance accumulators.

Input:  <feat_dir>/svd_accum.pt   dict[layer] = {XtX, sum, n}
Output: <out_dir>/svd_metrics.json, svd_spectrum.png, effective_rank_by_layer.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def spectrum_from_accum(XtX, s, n):
    """
    Given X^T X (D, D), sum-of-rows s (D,) and count n,
    return descending singular values of the centered matrix X - mean.
    """
    mu = s / n
    # centered Gram = XtX - n * mu mu^T
    C = XtX - n * torch.outer(mu, mu)
    # symmetrize to kill any roundoff asymmetry
    C = 0.5 * (C + C.T)
    eigvals = torch.linalg.eigvalsh(C).cpu().numpy().astype(np.float64)
    eigvals = np.clip(eigvals, 0, None)        # tiny negatives -> 0
    sigma = np.sqrt(np.sort(eigvals)[::-1])    # descending singular values
    return sigma


def rank_metrics(sigma):
    energy = sigma ** 2
    total = energy.sum()
    cum = np.cumsum(energy) / total
    stable_rank = float(total / (sigma[0] ** 2)) if sigma[0] > 0 else 0.0
    rank95 = int(np.searchsorted(cum, 0.95) + 1)
    rank99 = int(np.searchsorted(cum, 0.99) + 1)
    p = energy / total
    p_safe = np.clip(p, 1e-20, None)
    shannon = float(np.exp(-(p_safe * np.log(p_safe)).sum()))
    participation = float((sigma.sum() ** 2) / energy.sum())
    return {
        "stable_rank": stable_rank,
        "rank_at_95": rank95,
        "rank_at_99": rank99,
        "shannon_rank": shannon,
        "participation": participation,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accum = torch.load(feat_dir / "svd_accum.pt", map_location="cpu", weights_only=False)
    meta = json.loads((feat_dir / "meta.json").read_text())
    layers = sorted(accum.keys())
    D = meta["embed_dim"]

    metrics = {}
    spectra = {}
    for l in layers:
        a = accum[l]
        sigma = spectrum_from_accum(a["XtX"], a["sum"], a["n"])
        m = rank_metrics(sigma)
        metrics[l] = m
        spectra[l] = sigma
        print(f"layer {l:>2}: stable={m['stable_rank']:.1f} "
              f"rank95={m['rank_at_95']:<4} rank99={m['rank_at_99']:<4} "
              f"shannon={m['shannon_rank']:.1f}  D={D}")

    # ---- spectrum plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, l in enumerate(layers):
        s = spectra[l]
        ax.plot(np.arange(1, len(s) + 1), s / s[0],
                color=cmap(i / max(1, len(layers) - 1)),
                label=f"L{l}", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(f"per-layer SVD spectra — {meta['model']}")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "svd_spectrum.png", dpi=150)
    plt.close(fig)

    # ---- effective rank per layer ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color in [("stable_rank", "C0"),
                       ("rank_at_95", "C1"),
                       ("rank_at_99", "C2")]:
        ax.plot(layers, [metrics[l][key] for l in layers], "o-", color=color, label=key)
    ax.axhline(D, color="k", linestyle="--", alpha=0.5, label=f"d = {D}")
    ax.axhline(D / 2, color="k", linestyle=":", alpha=0.5, label=f"d/2 = {D // 2}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("rank")
    ax.set_title("effective rank vs layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_rank_by_layer.png", dpi=150)
    plt.close(fig)

    (out_dir / "svd_metrics.json").write_text(json.dumps(
        {"layers": layers, "embed_dim": D,
         "metrics": {str(l): metrics[l] for l in layers}}, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
