# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — budgeted, anchor-selective multi-source contextual interaction for 3D medical image segmentation.

[中文 README](README_CN.md)

---

> ### 📌 Code & architecture availability
>
> The **implementation**, **training/inference configs**, and **full architecture documentation** matching the accepted manuscript will be **released in this repository upon formal paper acceptance**.

---

## Overview

Dense global self-attention can supply explicit long-range context for 3D CT/MRI, but cost grows quadratically with tokens; windows/axes hard-code where interaction may occur, while token pruning often sparsifies the representation and then restores a dense pathway. **SPARK-UNet** takes the complementary route: **keep the dense multiscale convolutional feature grid intact** (local inductive bias and skips unchanged) and **sparsify computation, not representation**—spending a fixed interaction budget only at task-relevant locations.

Interaction factors into three controllable decisions: **where** (task-conditioned prior + PGTS), **how many** (stage-wise hard Top-K budget), and **what** (MSBA multi-source aggregation with gated residual write-back). Unselected sites keep the CNN base—no token completion or sparse-to-dense reprojection. TAE / WinMHSA3D add complementary axial-local and within-window modeling. Inserted as a switchable unit in nnU-Net, Spark isolates the interaction mechanism under matched protocols while preserving dense outputs and sliding-window inference.

---

## Pseudocode (product stack)

Default topology (**P5 → S6**, stages **E₀…E₅**): **E₀** TAE only; **E₁–E₄** Spark Block; **E₅** PriorHead on the CNN bottleneck; U-Net decoder with attention gates (AG); no bottleneck anchor. Symbols: TAE (tri-axial enhancement), PGTS (prior-gated token selection), MSBA (multi-source sparse block attention), WinMHSA3D (windowed MHSA).

### Algorithm 1 — Host forward (Spark–U-Net / Spark–ResEnc-UNet)

```text
Input:  volume X; encoder stages E0..E5 (P5→S6); class count C
Output: segmentation logits Y (deep supervision optional); training prior P_bot

hist ← ∅
for s = 0 .. 5 do                                # encoder stages
    F_s ← EncStage_s( F_{s-1} )                  # F_{-1} := X; CNN / ResEnc trunk
    if s ∈ {0} then                              # local-only: TAE
        H_s, Δ_s ← TAE(F_s)
        F_s ← H_s
        hist ← H_s
    else if s ∈ {1,2,3,4} then                   # guide: full Spark Block
        F_s, hist, P_eff ← SparkBlock_s(F_s, hist)
    # s = 5: CNN only; no Spark Block
end for

if training then
    P_bot ← PriorHead(F_5)                       # bottleneck prior (L_prior)
end if
Y ← DecoderAG(F_0..F_5)                          # AG before each skip concat
return Y [, P_bot]
```

### Algorithm 2 — Spark Block (Stages E₁–E₄)

```text
Input:  feature F; cascade hist H_prev (or ∅); budget K_s
Output: updated feature Out; cascade hist ≡ Out; prior map P_eff

# --- TAE ---
H, Δ ← TAE(F)                                    # H = F + mean(DW_d, DW_h, DW_w)
e ← Energy(Δ)                                    # per-sample residual energy

# --- task-conditioned prior ---
P_loc ← Softmax(1×1Conv(H))                      # C-class voxel prior
P* ← amplify_fg(P_loc; e)                        # fg channels × (1+γ·e)
P_eff ← clip(P* + λ·e·(1−fg(P*)))                # λ = 0.25 fallback

# --- PGTS: where + how many ---
I_raw ← σ(CNN_score(H))
I ← max(P_eff[1:]) · (ε + (1−ε)·I_raw)           # ε = 0.1; bg channel unused
Base ← H + 1×1(H ⊙ I)                            # zero-init refine
Idx ← TopK(I, K_s)                               # hard Top-K coords (non-diff.)

# --- MSBA: what + sparse write ---
Q ← Gather(Base, Idx)                            # K anchors as queries
HistTok ← Gather(Align(Proj(H_prev), grid), Idx) # self-hist if H_prev = ∅
G ← Linear(GAP(H))                               # global reference token
V_src ← {HistTok, broadcast(G), Q}               # multi-source KV
AttnTok ← Softmax_source(Q, V_src)               # normalize over sources, not N²
AttnTok ← Q + σ_tok([Q; AttnTok]) · (AttnTok−Q)  # per-anchor gated write
Cand ← Scatter(Base, Idx, AttnTok)               # unselected voxels unchanged
Out ← Base + σ_BA(H) · (Cand − Base)             # BaOnBase residual

# --- WinMHSA3D ∥ MSBA ---
Δ0 ← WinMHSA(H; shift=0) − H
Δs ← WinMHSA(H; shift=⌊W/2⌋) − H                 # dual-pass; W = window size
Out ← Out + σ_Win(H) · (Δ0 + Δs)

return Out, Out, P_eff
```

### Algorithm 3 — Token budget K

```text
# deep_stage = E4 (= max guide); deep_k_max = 16; k_floor = 32; k_cap = 512
# → (E1, E2, E3, E4) ≈ (128, 64, 32, 32)

K(s) ← clip( deep_k_max · 2^{E4−s} , [k_floor, k_cap] )
```

### Algorithm 4 — Decoder with attention gates

```text
Z ← F_5
for level ℓ = 4 .. 0 do
    Z ← Upsample(Z)
    Skip ← AG(g=Z, x=F_ℓ)                        # Attention U-Net gate
    Z ← ConvBlock( Concat(Z, Skip) )
    Y_ℓ ← SegHead(Z)                             # deep-supervision head if enabled
end for
return {Y_ℓ} or Y_0
```

**Complexity.** Dense CNN / TAE / PGTS scoring / WinMHSA act on the full feature grid. MSBA interaction is **O(K · S_src)** with S_src = #sources (hist + global ref + self), **not** O(N²). Unselected sites keep the convolutional base; no sparse-to-dense completion.

---

## Citation

If this work is relevant to your research, please cite the paper (BibTeX / DOI will be added here upon acceptance).

---

## 📜 License

The implementation will be released after acceptance, planned as **[Apache License 2.0](LICENSE.txt)**. Training/inference builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet); public benchmarks remain subject to their original terms.

---

## Data citation & downloads (nnU-Net format)

Cloud links below provide **nnU-Net–format** dataset packs. Official portals are for citation and licensing. Follow each dataset’s original terms of use.

### ACDC

| Type | Link |
|:-----|:-----|
| nnU-Net pack (Baidu Netdisk) | [Dataset100_ACDC.zip](https://pan.baidu.com/s/1AUUW16e-1RaTCaLbSaAgRA?pwd=3ufj) (code: `3ufj`) |
| nnU-Net pack (Kaggle) | [acdc-datasat](https://www.kaggle.com/datasets/shuaishuaichai/acdc-datasat) |
| Official source | [Human Heart Project / ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| TransUNet-split preprocessed reference | [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4) |

### Synapse / BTCV

| Type | Link |
|:-----|:-----|
| nnU-Net pack (Baidu Netdisk) | [Dataset180_Synapse.zip](https://pan.baidu.com/s/17yYXaYWeLJQxMA6Txyl15w?pwd=idkd) (code: `idkd`) |
| nnU-Net pack (Kaggle) | [synapse-dataset](https://www.kaggle.com/datasets/shuaishuaichai/synapse-dataset) |
| Official source (BTCV / Synapse) | [Synapse: syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| TransUNet-split preprocessed reference | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

### BraTS 2021 Adult Glioma

Paper and experiments use the **BraTS 2021** publicly labeled training cohort (1251 cases). BraTS 2022/2023 Adult Glioma redistributed the same cohort and are **not** BraTS 2025 Lighthouse. The Baidu pack is named `Dataset1251_BraTS2023GLI.zip` and matches that same cohort.

| Type | Link |
|:-----|:-----|
| nnU-Net pack (Baidu Netdisk) | [Dataset1251_BraTS2023GLI.zip](https://pan.baidu.com/s/1Ji6Uh6g_Fs6lDOMHkFaktw?pwd=vz41) (code: `vz41`) |
| nnU-Net pack (Kaggle) | [brats2021-dataset](https://www.kaggle.com/datasets/shuaishuaichai/brats2021-dataset) |
| Official source (BraTS 2021) | [Synapse: syn25829067](https://www.synapse.org/Synapse:syn25829067) |
| Same-cohort redistribute page (2023 challenge) | [Synapse: syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351) |
