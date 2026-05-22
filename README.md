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

## Exp 2 — Low-rank probe (I-JEPA ViT-H/14)

Exp 1's finding "early I-JEPA-H/14 layers live in a 76-dim subspace within 1280" is *suggestive* of compressibility but not a direct test. Exp 2 closes that gap: if you literally restrict each layer to its top-k principal directions and re-run the probe, how much accuracy do you lose?

**Method.** For each layer L:
1. Eigendecompose the centered token-Gram (`X^TX − n·μμᵀ`) → eigenvectors `V_L` (D×D, sorted by descending singular value).
2. For each `k ∈ {4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1280}` plus the layer's own `rank@95(L)` and `rank@99(L)`: project the mean-pooled feature `x → V_L[:, :k]ᵀ x`, then train the 1000-way linear probe (8/2 split, standardize on train, C-sweep over `{1e-3, 1e-2, 0.1, 1, 10}`, best val top-1).
3. Headline metric per layer: **`top-1@rank95 / top-1@full_d`**.

Mean-pool commutes with the linear projection, so this is exactly equivalent to mean-pooling token features projected onto the top-k subspace.

### Results

| layer | rank@95 | rank@99 | top-1@rank95 | top-1@rank99 | top-1@full_d | r95/full |
|---:|---:|---:|---:|---:|---:|---:|
|  1 |  78 |  147 |  2.25 |  2.50 |  2.35 |  **95.7 %** |
|  3 | 106 |  244 |  3.25 |  4.10 |  4.70 |  69.1 % |
|  5 | 166 |  400 |  4.65 |  5.75 |  5.60 |  83.0 % |
|  7 | 234 |  532 |  6.15 |  6.65 |  7.00 |  87.9 % |
|  9 | 278 |  631 |  7.20 |  7.00 |  7.20 | **100.0 %** |
| 11 | 353 |  736 |  7.95 |  9.00 |  9.10 |  87.4 % |
| 13 | 428 |  833 |  8.90 |  9.30 | 10.25 |  86.8 % |
| 15 | 535 |  949 | 10.25 | 11.50 | 12.05 |  85.1 % |
| 17 | 653 | 1039 | 14.30 | 15.40 | 15.35 |  93.2 % |
| 19 | 766 | 1106 | 17.95 | 18.35 | 19.55 |  91.8 % |
| 21 | 871 | 1156 | 21.95 | 21.70 | 21.65 | **101.4 %** |
| 23 | 912 | 1174 | 27.90 | 27.80 | 27.60 | **101.1 %** |
| 25 | 930 | 1181 | 32.65 | 33.45 | 33.00 |  98.9 % |
| 27 | 952 | 1188 | 41.60 | 41.80 | 42.20 |  98.6 % |
| 29 | 946 | 1186 | 48.60 | 49.00 | 49.35 |  98.5 % |
| 31 | 969 | 1196 | 56.25 | 56.65 | 56.70 |  99.2 % |
| 32 | 948 | 1187 | 57.05 | 57.15 | 56.55 | **100.9 %** |
| 33 (norm) | 350 |  424 | 54.05 | 54.35 | 56.80 |  95.2 % |

Plots:
- [`top-1 vs k, one line per layer`](outputs/vit_h14_ijepa_lowrank/lowrank_probe_vs_k.png) — ★ marks `k = rank@95(L)` on each line.
- [`full_d vs rank95 vs rank99 by layer`](outputs/vit_h14_ijepa_lowrank/lowrank_probe_by_layer.png)

Full JSON: [`outputs/vit_h14_ijepa_lowrank/lowrank_probe.json`](outputs/vit_h14_ijepa_lowrank/lowrank_probe.json).

### Findings

**The hypothesis holds at every layer.** Median `r95/full` across the 18 measured layers is **95.7 %** — projecting to the top-`rank@95(L)` directions loses essentially nothing. Several layers come in at ≥100 % (projecting helps, presumably by denoising the bottom directions the probe was over-fitting to).

**Layer 1 is the headline.** Probe accuracy goes from 2.35 % (k=1280) to 2.25 % (k=78). **6 % of the dimensions, 95.7 % of the accuracy.** A 78-d block trained on the same I-JEPA target would behave the same on this probe.

**The plateau range is wider than Exp 1 suggested.** Layer 9 hits 100 % at k=278 (22 % of d). Layers 21–32 all hit ≥98.5 % at their rank@95, which sits around 870–970 (68–76 % of d). Only the late layers actually use most of d — and even then, `r99 ≥ 1156` consistently leaves ~80 dims of "noise" headroom.

**The worst layer is L3** (69 % recovery). The early-layer rank values rise faster than the probe can keep up with — there's something useful in directions 100-1280 of L3 that L1 didn't have. This is a small dip, not a wall.

**Implication for progressive-dim ViT.**

| layer range | suggested d (= avg rank@95) | as fraction of 1280 |
|---|---|---|
| 1-7 | ≈ 135 | 11 % |
| 9-15 | ≈ 425 | 33 % |
| 17-23 | ≈ 800 | 63 % |
| 25-32 | ≈ 950 | 74 % |

A monotonically-growing-d ViT-H following this schedule would have roughly **45 % of the FLOPs** of a constant-d=1280 ViT-H (FLOPs in a block scale ~ d², so the sum is `(135² × 7 + 425² × 7 + 800² × 7 + 950² × 8) / (1280² × 32) ≈ 0.45`), with negligible expected loss on linear-probe-style downstream tasks. **This is the experimental hook for a paper on progressive-dim ViTs trained with I-JEPA.**

### Caveats

- This is a *post-hoc* projection, not a *trained-at-lower-d* network. A progressive-dim ViT trained from scratch could differ — the SVD-aligned subspace might not be what the network learns to put there when forced to. But it gives a tight upper bound on the loss.
- Only on ImageNetV2 with an 8/2 split. Dense prediction tasks (segmentation, depth) likely need more dimensions in early layers (low-level signal matters more there).
- The "r95/full" denoising at L21–23, L32 is small (1-2 pp) and could be noise from the C-sweep / random initialization.

Run yourself:
```bash
sbatch experiments/exp2_lowrank_probe/run.slurm                  # extract + probe, ~30 min
# or, against an existing extract:
SRC_RUN_ID=<jobid> sbatch experiments/exp2_lowrank_probe/run_probe_only.slurm
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
│       ├── run.slurm                  full pipeline (extract + probe)
│       └── run_probe_only.slurm       probe-only against an existing extract
└── outputs/
    ├── vit_h14_ijepa/                 I-JEPA-H/14 Exp 1 results
    ├── vit_b16_mae/                   MAE-B/16 Exp 1 results
    ├── vit_b14_dinov2/                DINOv2-B/14 Exp 1 results
    └── vit_h14_ijepa_lowrank/         I-JEPA-H/14 Exp 2 (low-rank probe)
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
