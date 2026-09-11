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

交互被拆成三个可独立控制的决策：**where**（任务条件先验 + PGTS）、**how many**（逐 stage 硬 Top-K 预算）、**what**（MSBA 多源聚合 + 门控残差写回）；未选中位置保持 CNN 基底，无需 token 补全或稀疏→稠密重投影。TAE / WinMHSA3D 补充轴向局部与窗内建模。模块作为 nnU-Net 中可开关单元插入，便于在匹配协议下隔离交互机制本身的贡献，并继续支持稠密输出与滑窗推理。

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
