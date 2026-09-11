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
