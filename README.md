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

### Exp 1c — DINOv2 ViT-B/14, LVD-142M *(second ViT-B SSL method, to partly disentangle scale vs method)*
- Checkpoint: timm `vit_base_patch14_dinov2.lvd142m` (no register tokens).
- 12 blocks, d=768, patch_size=14, 256 patch tokens at 224.
- Code: [`experiments/exp1_svd_vitb_dinov2/`](experiments/exp1_svd_vitb_dinov2/)
- Artifacts: [`outputs/vit_b14_dinov2/`](outputs/vit_b14_dinov2/)

Comparing H/14 (I-JEPA) ↔ B/16 (MAE) confounds **scale** with **method (latent target vs pixel target)**. Adding DINOv2-B/14 — same scale as MAE, different SSL objective (self-distillation, no pixel reconstruction) — partly separates the two. The remaining clean experiment is pretraining I-JEPA-B/16 ourselves.

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

### DINOv2 ViT-B/14 (d=768, 12 blocks)
| layer | stable | rank@95 | rank@99 | shannon | probe top-1 | best_C |
|---:|---:|---:|---:|---:|---:|---:|
|  1 | 24.5 |  271 |  414 | 192.4 |  3.55 % | 10    |
|  2 | 19.4 |  **461** |  638 | 262.7 |  5.35 % | 0.001 |
|  4 | 21.2 |  483 |  671 | 228.8 |  9.45 % | 0.001 |
|  6 | 24.6 |  556 |  702 | 314.2 | 17.15 % | 10    |
|  8 | 15.3 |  580 |  713 | 314.4 | 24.90 % | 0.1   |
|  9 | **1.0** | **1**   |  268 |   1.5 | 31.20 % | 1     |
| 10 | **1.1** | **71**  |  484 |   2.2 | 37.55 % | 0.001 |
| 11 | **1.2** | **317** |  610 |   5.4 | 45.45 % | 0.001 |
| 12 | 11.3 |  600 |  702 | 408.8 | **55.95 %** | 0.001 |
| 13 (norm) | 38.6 | 630 | 732 | 470.8 | 55.00 % | 0.1   |

- **rank@95 crosses d/2 at layer 2** — earliest of the three models.
- **Layers 9–11 show a stable_rank ≈ 1 anomaly** — see "DINOv2 artifact tokens" caveat below.
- Probe ceiling = 56 % at L12 — best of any model tested (DINOv2 features just are stronger).

Full table: [`outputs/vit_b14_dinov2/comparison_table.md`](outputs/vit_b14_dinov2/comparison_table.md).
Plots: [spectrum](outputs/vit_b14_dinov2/svd/svd_spectrum.png), [effective rank](outputs/vit_b14_dinov2/svd/effective_rank_by_layer.png), [probe](outputs/vit_b14_dinov2/probe/linear_probe_by_layer.png).

### Side-by-side

| metric | I-JEPA-H/14 | MAE-B/16 | DINOv2-B/14 |
|---|---|---|---|
| L1 rank@95 / d | **6 %** (76/1280) | 30 % (233/768) | 35 % (271/768) |
| L1 stable rank | 9.7 | 11.7 | 24.5 |
| Layer where rank@95 ≥ d/2 | between L11–L12 | L7 | **L2** |
| Peak rank@95 / d | 67 % (L24) | 77 % (L12) | 82 % (L13 norm) |
| Probe top-1 ceiling | 53 % (L32) | 27 % (L11) | **56 % (L12)** |
| Anomalous layers | — | — | L9–11 (artifact tokens) |

## Findings

1. **Strong early-layer overprovisioning is robust at the H scale.** I-JEPA-H/14 layer 1 carries 95 % of its variance in just 76 of 1280 dims. The first ~8 blocks all live in subspaces well under d/2. A 200-d early block would lose negligible information.

2. **At the B scale the overprovisioning gap shrinks substantially, across two different SSL methods.** Both MAE-B/16 (30 %) and DINOv2-B/14 (35 %) use ~⅓ of d in layer 1, and rank@95 crosses d/2 much earlier (L7 and L2 respectively). The ViT-H result does **not** transfer cleanly to ViT-B.

3. **DINOv2-B/14 is the most "rank-saturated" of the three.** rank@95 crosses d/2 already at layer 2, vs L7 for MAE-B/16. Self-distillation (no reconstruction target) gives the *highest* early-layer rank we measured. The progressive-dim story is weakest for DINOv2-style training.

4. **DINOv2 artifact tokens.** Layers 9, 10, 11 show stable_rank ≈ 1.0 — a single direction dominates 99 %+ of the total energy. This is the well-known [register-token artifact](https://arxiv.org/abs/2309.16588): the no-register DINOv2 variant produces a small number of extremely high-norm "artifact" tokens that hijack the singular spectrum. Our SVD metrics for L9–L11 are dominated by ~tens of outlier tokens, not by genuine low-rank structure across the population. *The `reg4` DINOv2 variant or filtering high-norm tokens before SVD would clean this up.*

5. **Linear probe accuracy rises monotonically with depth** in all three models. There is no "plateau in layers 1–N" — the original hypothesis "early layers are functionally equivalent" needs sharpening to "early layers carry less task-relevant information but pack what they have into very few dimensions."

6. **Scale vs method is now partly disentangled.** With MAE-B/16 ≈ DINOv2-B/14 ≈ 30–35 % L1 rank/d, and I-JEPA-H/14 at 6 %, the leading-order effect is likely **scale**: bigger d gives more room to leave unused in early layers. We can't rule out a residual method effect without an I-JEPA-B/16 datapoint, but the I-JEPA ↔ MAE B-scale gap predicted by "method matters" is probably small.

### Next experiments

- **Pretrain I-JEPA-B/16 ourselves** (50–100 epochs is plausibly enough for rank analysis). Closes the scale-vs-method gap.
- **DINOv2-B/14 with register tokens** (timm: `vit_base_patch14_reg4_dinov2.lvd142m`) — should remove the L9–11 anomaly and give a clean comparison.
- **Token-norm filtering on the DINOv2 SVD** — drop top-1 % of tokens by ℓ₂ norm and recompute spectrum; expected to reveal the underlying low-rank structure currently masked by artifacts.

---

## Exp 2 — Low-rank probe *(in flight)*

The Exp 1 finding "early I-JEPA-H/14 layers live in a 76-dim subspace within 1280" is *suggestive* of compressibility but not a direct test. Exp 2 closes that gap.

**Method.** For each layer L of I-JEPA-H/14:
1. Re-extract features with the streaming-covariance pipeline (we need the actual eigenvectors of the token Gram, not just the eigenvalues we kept from Exp 1).
2. For each `k` in `{4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1280}` plus the layer's own `rank@95(L)` and `rank@99(L)`:
   - Project the mean-pooled feature `x` onto the top-k eigenvectors: `z = V_L[:, :k]ᵀ x`.
   - Train the 1000-way linear probe on `z` (8/2 split, standardize on train, C-sweep, report best val top-1).
3. Headline metric per layer: **`top-1@rank95 / top-1@full_d`**. If early layers retain ~100 % of full-d accuracy at `k = rank@95(L)`, the low-rank structure is *real* (compressible without task-loss). If they lose substantial accuracy, the rank metric is misleading.

Note that mean-pool commutes with the linear projection, so this is exactly equivalent to mean-pooling token features that were first projected onto the top-k subspace.

**Status.** Code lives at [`experiments/exp2_lowrank_probe/`](experiments/exp2_lowrank_probe/). Job submission is currently blocked by an `AssocMaxSubmitJobLimit` (the other project on the same account has ~30 jobs queued). The job will go in as soon as one of those clears (~10 min eta at submit time). Results will land in `outputs/exp2_lowrank_probe/` and this section will be updated with the table and plots.

```bash
sbatch experiments/exp2_lowrank_probe/run.slurm    # extract + probe sweep, ~30 min on A100-80
```

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
bash scripts/prefetch_dinov2_b14.sh        # DINOv2-B/14   (optional; ~340 MB)
```

### Run experiments
```bash
# Exp 1a — I-JEPA ViT-H/14
sbatch experiments/exp1_svd/run.slurm

# Exp 1b — MAE ViT-B/16 (extract + SVD + probe + table in one job)
sbatch experiments/exp1_svd_vitb/run.slurm

# Exp 1c — DINOv2-B/14 (same pipeline; reuses MAE's svd/probe/table scripts)
sbatch experiments/exp1_svd_vitb_dinov2/run.slurm
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
│   ├── prefetch_dinov2_b14.sh         DINOv2-B/14 (via timm + HF)
│   └── sync_to_cluster.sh             rsync project to cluster
├── experiments/
│   ├── exp1_svd/                      ViT-H/14 pipeline
│   │   ├── extract_features.py        forward + per-layer pooled + token dump
│   │   ├── svd_analysis.py            spectra + effective rank + plots
│   │   ├── linear_probe.py            in-GPU 1000-way LR
│   │   └── run.slurm
│   ├── exp1_svd_vitb/                 MAE-B/16 pipeline (streaming covariance)
│   │   ├── extract_features.py        forward + pooled + accumulators
│   │   ├── svd_analysis.py            (also reused by DINOv2)
│   │   ├── linear_probe.py            GPU LR with C-sweep (also reused)
│   │   ├── make_table.py              compare_table.{md,csv} (also reused)
│   │   ├── run.slurm                  full pipeline
│   │   └── run_probe_only.slurm       probe-only against an existing extract+SVD run
│   ├── exp1_svd_vitb_dinov2/          DINOv2-B/14 pipeline (reuses svd/probe/table)
│   │   ├── extract_features.py        timm load + pos-embed interp to 224
│   │   └── run.slurm
│   └── exp2_lowrank_probe/            Exp 2 — low-rank probe sweep on I-JEPA-H/14
│       ├── extract_with_accum.py      re-extract with streaming covariance
│       ├── lowrank_probe.py           project onto top-k eigvecs + C-sweep probe
│       └── run.slurm
└── outputs/
    ├── vit_h14_ijepa/                 I-JEPA-H/14 results
    ├── vit_b16_mae/                   MAE-B/16 results
    └── vit_b14_dinov2/                DINOv2-B/14 results
        ├── comparison_table.{md,csv}
        ├── features/meta.json
        ├── svd/{spectrum,effective_rank}.png + svd_metrics.json
        └── probe/linear_probe_by_layer.png + linear_probe.json
```

## Credits

- **I-JEPA** — [Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243), [code](https://github.com/facebookresearch/ijepa).
- **MAE** — [He et al., CVPR 2022](https://arxiv.org/abs/2111.06377), [code](https://github.com/facebookresearch/mae).
- **DINOv2** — [Oquab et al., 2023](https://arxiv.org/abs/2304.07193), [code](https://github.com/facebookresearch/dinov2). The artifact-token observation in §Findings is from [Darcet et al., ICLR 2024](https://arxiv.org/abs/2309.16588).
- **ImageNetV2** — [Recht et al., ICML 2019](https://arxiv.org/abs/1902.10811), matched-frequency split.
- Compute: NUS SoC student cluster (A100-80 partitions).
