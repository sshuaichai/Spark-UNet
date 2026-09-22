# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — 预算约束的锚点选择性多源上下文交互，用于三维医学图像分割。

[English README](README_EN.md)

---

> ### 📌 代码与架构公开说明
>
> 与正式稿件一致的可复现 **实现代码**、**训练/推理配置** 及 **完整架构说明**，将在论文 **被期刊/会议正式接受（acceptance）后** 于本仓库公开。

---

## 方法概览

稠密全局自注意力能为 3D CT/MRI 提供显式长程上下文，但代价随 token 数二次增长；窗口/轴向等方案又把交互位置写死，token 剪枝则往往先稀疏表征再补回稠密通路。**SPARK-UNet** 走互补路线：**保留完整的多尺度卷积特征网格**（局部归纳偏置与 skip 不变），**稀疏的是计算而非表征**——在任务相关位置上按固定交互预算注入上下文。

交互被拆成三个可独立控制的决策：**where**（任务条件先验 + PGTS）、**how many**（逐 stage 硬 Top-K 预算）、**what**（MSBA 多源聚合 + 门控残差写回）；未选中位置保持 CNN 基底，无需 token 补全或稀疏→稠密重投影。同一 Spark Block 装在两种卷积基座上：**Spark–U-Net** 与 **Spark–ResEnc-UNet**。基座保留局部卷积。TAE 沿轴向细化各引导阶段，WinMHSA3D 是模块内的窗口支路，与 MSBA 并行。默认排序为 **Lock-Pr**：先验与可学习分数共用一个硬 Top-K 表。训练与比较使用匹配的 3d fullres 协议；输出保持稠密，推理保持滑窗。

---

## 伪代码（产品默认栈）

默认拓扑（**P5 → S6**，阶段 **E₀…E₅**）：**E₀** 仅 TAE；**E₁–E₄** Spark Block；**E₅** 瓶颈 PriorHead；带注意力门（AG）的 U-Net 解码器；无瓶颈锚点。符号：TAE（三轴向增强）、PGTS（先验门控 token 选择）、MSBA（多源稀疏块注意力）、WinMHSA3D（窗口 MHSA）。

### 算法 1 — 主机前向（Spark–U-Net / Spark–ResEnc-UNet）

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

### 算法 2 — Spark Block（Stages E₁–E₄）

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

### 算法 3 — Token 预算 K

```text
# deep_stage = E4 (= max guide); deep_k_max = 16; k_floor = 32; k_cap = 512
# → (E1, E2, E3, E4) ≈ (128, 64, 32, 32)

K(s) ← clip( deep_k_max · 2^{E4−s} , [k_floor, k_cap] )
```

### 算法 4 — 带注意力门的解码器

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

**复杂度。** 稠密 CNN / TAE / PGTS 打分 / WinMHSA 在完整特征网格上运行。MSBA 交互代价为 **O(K · S_src)**（S_src = 源数：hist + 全局参考 + self），**不是** O(N²)。未选中位点保留卷积基底，无需稀疏→稠密补全。

---

## 引用

若本工作与您的研究相关，请引用正式论文（接受后将在此补充 BibTeX / DOI）。

---

## 📜 License

方法实现将于接受后发布，计划以 **[Apache License 2.0](LICENSE.txt)** 授权。训练/推理基于 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)；公开数据集须遵守各自使用条款。

---

## 数据引用与下载（nnU-Net 格式）

下列网盘提供 **已转换为 nnU-Net 格式** 的数据包；官方源站用于引用与授权。使用前请遵守各数据集原始条款。

### ACDC

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [Dataset100_ACDC.zip](https://pan.baidu.com/s/1AUUW16e-1RaTCaLbSaAgRA?pwd=3ufj)（提取码：`3ufj`） |
| nnU-Net 格式（Kaggle） | [acdc-datasat](https://www.kaggle.com/datasets/shuaishuaichai/acdc-datasat) |
| 官方源 | [Human Heart Project / ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| TransUNet 划分预处理参考 | [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4) |

### Synapse / BTCV

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [Dataset180_Synapse.zip](https://pan.baidu.com/s/17yYXaYWeLJQxMA6Txyl15w?pwd=idkd)（提取码：`idkd`） |
| nnU-Net 格式（Kaggle） | [synapse-dataset](https://www.kaggle.com/datasets/shuaishuaichai/synapse-dataset) |
| 官方源（BTCV / Synapse） | [Synapse: syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| TransUNet 划分预处理参考 | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

### BraTS 2021 Adult Glioma

论文与实验口径为 **BraTS 2021** 公开带标注训练队列（1251 例）。BraTS 2022/2023 Adult Glioma 为同一队列的再分发，**不等于** BraTS 2025 Lighthouse。百度包文件名为 `Dataset1251_BraTS2023GLI.zip`，与上述同队列一致。

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [Dataset1251_BraTS2023GLI.zip](https://pan.baidu.com/s/1Ji6Uh6g_Fs6lDOMHkFaktw?pwd=vz41)（提取码：`vz41`） |
| nnU-Net 格式（Kaggle） | [brats2021-dataset](https://www.kaggle.com/datasets/shuaishuaichai/brats2021-dataset) |
| 官方源（BraTS 2021） | [Synapse: syn25829067](https://www.synapse.org/Synapse:syn25829067) |
| 同队列再分发参考（2023 challenge 页） | [Synapse: syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351) |
