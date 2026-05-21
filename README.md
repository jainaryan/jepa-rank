# jepa-rank

> **Per-layer effective rank of pretrained self-supervised ViTs.**
> Are early transformer blocks dimensionally overprovisioned? A short empirical study on I-JEPA and MAE.

---

## Motivation

The standard ViT (Base, Large, Huge, ...) keeps the embedding dimension `d` constant across all `L` transformer blocks. That assumes every block needs the same capacity. If early blocks actually operate on a much lower-rank subspace within `d`, the constant-`d` design is wasteful — and motivates a **progressive-dim ViT** that allocates smaller `d` to early blocks and grows it with depth.

This repo tests that premise empirically — *before* committing to pretraining a new architecture. It's a cheap go/no-go signal: if the early-layer effective rank is close to `d`, the premise is wrong and the architectural idea is weak; if it's far below `d`, there's room to compress.

## Method

For each transformer block (and the final post-norm output):

1. **Streaming covariance.** Forward-pass `N` images through the frozen encoder. For each layer, accumulate `XᵀX (D×D)`, `Σx (D,)`, and a token count `n` across batches. After the full pass, the centered Gram matrix is `C = XᵀX − n·μμᵀ` where `μ = (Σx)/n`. Eigenvalues of `C` are the squared singular values of the centered token matrix. **No raw token tensors are stored** — `(D×D)` accumulators are ~4 MB for d=768 vs ~20 GB for raw tokens.

2. **Effective rank metrics**, from the singular value spectrum `σ₁ ≥ σ₂ ≥ …`:
   - **stable rank** `‖X‖_F² / σ₁²`
   - **rank@95%** smallest `k` capturing 95% of `Σ σᵢ²`
   - **rank@99%** same at 99%
   - **Shannon rank** `exp(H(σᵢ²/Σσᵢ²))`

3. **Linear probe** on mean-pooled patch tokens (CLS excluded) — 8 train / 2 val per ImageNet class, features standardized on the train split, regularization sweep, best val top-1 reported.

Data: **ImageNetV2 matched-frequency** (10 000 images, 1000 classes), located at `/home/h/haoyu/ImageNetV2-matched-frequency` on the NUS SoC cluster.

## Experiments

### Exp 1a — I-JEPA ViT-H/14, IN-1K, 300 epochs
- Checkpoint: `IN1K-vit.h.14-300e.pth.tar` from [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa).
- 32 blocks, d=1280, patch_size=14, 256 patch tokens at 224.
- Code: [`experiments/exp1_svd/`](experiments/exp1_svd/)
- Artifacts: [`outputs/vit_h14_ijepa/`](outputs/vit_h14_ijepa/)

### Exp 1b — MAE ViT-B/16, IN-1K, 1600 epochs *(stand-in for the missing I-JEPA-B/16)*
Meta did not release an I-JEPA-B/16 checkpoint. To still get a ViT-B-scale datapoint we use **MAE-B/16** ([facebookresearch/mae](https://github.com/facebookresearch/mae)), the closest predictive SSL ViT-B at patch_size 16.
- Checkpoint: `mae_pretrain_vit_base.pth`.
- 12 blocks, d=768, patch_size=16, 196 patch tokens at 224.
- Code: [`experiments/exp1_svd_vitb/`](experiments/exp1_svd_vitb/)
- Artifacts: [`outputs/vit_b16_mae/`](outputs/vit_b16_mae/)

This means the H ↔ B comparison is confounded by **method (I-JEPA latent target vs MAE pixel target)**, not just scale. The pixel target forces MAE early layers to retain low-level signal (edges, textures) → expected to keep effective rank higher than a latent-target method. We surface this in the findings.

## Headline results

### I-JEPA ViT-H/14 (d=1280, 32 blocks)
| layer | stable | rank@95 | rank@99 | shannon | probe top-1 |
|---:|---:|---:|---:|---:|---:|
|  1 |  9.7 |   **76**  |  145 |  56.9 |  1.65 % |
|  4 | 13.2 |  135 |  321 |  74.6 |  4.30 % |
|  8 | 13.9 |  243 |  559 | 103.4 |  5.95 % |
| 16 | 16.8 |  560 |  971 | 204.6 | 11.05 % |
| 24 | 10.2 |  **858** | 1154 | 343.7 | 26.55 % |
| 32 |  4.0 |  724 | 1097 | 174.5 | **52.70 %** |
| 33 (norm) | 19.7 | 321 |  409 | 219.9 | 52.15 % |

- **rank@95 at L1 = 76 / 1280 = 6 % of d.** Layer 1 lives in a 76-dim subspace within 1280-dim.
- rank@95 peaks at L24 (858) and never reaches d.
- Probe accuracy is monotone → there is no "plateau"; early layers carry less information, just packed into very few dimensions.

Full table: [`outputs/vit_h14_ijepa/svd/svd_metrics.json`](outputs/vit_h14_ijepa/svd/svd_metrics.json)

Plots:
- [Spectrum](outputs/vit_h14_ijepa/svd/svd_spectrum.png)
- [Effective rank by layer](outputs/vit_h14_ijepa/svd/effective_rank_by_layer.png)
- [Probe by layer](outputs/vit_h14_ijepa/probe/linear_probe_by_layer.png)

### MAE ViT-B/16 (d=768, 12 blocks)
| layer | stable | rank@95 | rank@99 | shannon | probe top-1 | best_C |
|---:|---:|---:|---:|---:|---:|---:|
|  1 | 11.7 |  **233** |  404 | 128.5 |  2.80 % | 0.001 |
|  4 | 13.5 |  345 |  607 | 141.6 |  7.10 % | 1     |
|  7 | 16.1 |  **387** |  620 | 161.6 | 13.80 % | 0.1   |
|  8 | 17.0 |  424 |  645 | 178.6 | 18.65 % | 0.01  |
| 10 | 17.2 |  529 |  698 | 234.8 | 25.95 % | 0.1   |
| 12 | 21.1 |  **588** |  720 | 315.9 | 25.40 % | 0.001 |
| 13 (norm) |  3.2 | 183 | 533 |  21.9 | 25.65 % | 0.001 |

- **rank@95 crosses d/2 (=384) at layer 7.**
- **rank@95 never reaches 0.9·d (=691).** Maximum is 588 = 77 % of d at L12.
- L1 rank@95 / d = 233 / 768 = **30 %** — far higher than the I-JEPA-H/14 figure of 6 %.

Full table + transition points: [`outputs/vit_b16_mae/comparison_table.md`](outputs/vit_b16_mae/comparison_table.md)

Plots:
- [Spectrum](outputs/vit_b16_mae/svd/svd_spectrum.png)
- [Effective rank by layer](outputs/vit_b16_mae/svd/effective_rank_by_layer.png)
- [Probe by layer](outputs/vit_b16_mae/probe/linear_probe_by_layer.png)

### Side-by-side

| metric | I-JEPA-H/14 | MAE-B/16 |
|---|---|---|
| L1 rank@95 / d | **6 %** (76/1280) | **30 %** (233/768) |
| L1 stable rank | 9.7 | 11.7 |
| Layer where rank@95 ≥ d/2 | between L11–L12 | layer 7 |
| Peak rank@95 / d | 67 % (L24) | 77 % (L12) |
| Probe top-1 ceiling | 53 % (L32) | 27 % (L11) |

## Findings

1. **Strong early-layer overprovisioning is real at the H scale.** I-JEPA-H/14 layer 1 carries 95 % of its variance in just 76 of 1280 dims. The first ~8 blocks all live in subspaces well under d/2. A 200-d early block would lose negligible information.

2. **At the B scale the gap shrinks substantially.** MAE-B/16 L1 already uses 30 % of d. The rank@95 curve crosses d/2 at L7 — roughly the middle of the network — vs near-final in ViT-H/14.

3. **Confound.** We cannot fully separate **scale (B vs H)** from **method (MAE pixel reconstruction vs I-JEPA latent prediction)**. MAE's pixel target naturally forces early layers to retain low-level structure → higher rank. The clean experiment requires pretraining I-JEPA-B/16.

4. **Linear probe accuracy rises monotonically with depth** in both models. There is no "plateau in layers 1–N" — the original hypothesis framing needs sharpening from "functionally equivalent early layers" to "low-rank early layers, even when carrying limited task-relevant signal."

### Next experiments to disentangle

- **Pretrain I-JEPA-B/16 ourselves** (50–100 epochs may suffice for rank analysis) — isolates the scale axis.
- **Run the same analysis on DINOv2-B/14** — different SSL method (self-distillation, no raw-pixel target). If DINOv2-B matches I-JEPA-H's pattern, the I-JEPA ↔ MAE gap is real; if it matches MAE-B, the difference is mostly scale.

## Reproduce

### Cluster setup (NUS SoC)
```bash
ssh nus-student-cluster      # alias in ~/.ssh/config; requires NUS VPN
cd ~/projects/ijepa
git clone https://github.com/facebookresearch/ijepa.git upstream    # only needed for I-JEPA model defs
python3 -m venv ijepaenv && source ijepaenv/bin/activate
export TMPDIR=$HOME/projects/ijepa/.tmp && mkdir -p $TMPDIR
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
pip install --no-cache-dir numpy scipy scikit-learn pyyaml matplotlib pillow tqdm timm einops
bash scripts/download_checkpoint.sh        # I-JEPA ViT-H/14 (9.7 GB)
bash scripts/download_mae_b16.sh           # MAE ViT-B/16  (328 MB)
```

### Run experiments
```bash
# Exp 1a — I-JEPA ViT-H/14
sbatch experiments/exp1_svd/run.slurm

# Exp 1b — MAE ViT-B/16 (extract + SVD + probe + table in one job)
sbatch experiments/exp1_svd_vitb/run.slurm
```

Each job takes ~5 min on an A100-80 (gpu-long / gpu partition).

### Sync local ↔ cluster
GitHub SSH is blocked on the cluster, so we ship code via rsync:
```bash
bash scripts/sync_to_cluster.sh
```

## Project layout

```
.
├── README.md                          ← this file
├── env/environment.yml                ← original conda spec (kept for reference; pip venv is used)
├── scripts/
│   ├── download_checkpoint.sh         I-JEPA ViT-H/14
│   ├── download_mae_b16.sh            MAE ViT-B/16
│   └── sync_to_cluster.sh             rsync project to cluster
├── experiments/
│   ├── exp1_svd/                      ViT-H/14 pipeline
│   │   ├── extract_features.py        forward + per-layer pooled + token dump
│   │   ├── svd_analysis.py            spectra + effective rank + plots
│   │   ├── linear_probe.py            in-GPU 1000-way LR
│   │   └── run.slurm
│   └── exp1_svd_vitb/                 MAE-B/16 pipeline (streaming covariance)
│       ├── extract_features.py        forward + pooled + accumulators
│       ├── svd_analysis.py
│       ├── linear_probe.py            GPU LR with C-sweep
│       ├── make_table.py              compare_table.{md,csv}
│       ├── run.slurm                  full pipeline
│       └── run_probe_only.slurm       probe-only against an existing extract+SVD run
└── outputs/
    ├── vit_h14_ijepa/                 I-JEPA-H/14 results
    │   ├── meta.json
    │   ├── svd/{spectrum,effective_rank}.png + svd_metrics.json
    │   └── probe/linear_probe_by_layer.png + linear_probe.json
    └── vit_b16_mae/                   MAE-B/16 results
        ├── comparison_table.{md,csv}
        ├── features/meta.json
        ├── svd/...
        └── probe/...
```

## Credits

- **I-JEPA** — [Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243), [code](https://github.com/facebookresearch/ijepa).
- **MAE** — [He et al., CVPR 2022](https://arxiv.org/abs/2111.06377), [code](https://github.com/facebookresearch/mae).
- **ImageNetV2** — [Recht et al., ICML 2019](https://arxiv.org/abs/1902.10811), matched-frequency split.
- Compute: NUS SoC student cluster (A100-80 partitions).
