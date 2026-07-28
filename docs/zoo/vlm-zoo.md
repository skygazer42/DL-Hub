---
icon: material/image-text
---

# VLM Zoo

> **70 算法族 / 210 Architecture IDs** --- 覆盖 2021-2025 年视觉-语言多模态模型的核心演进路线，从对比学习、指令微调到文档理解与端侧多模态（每族 `tiny/small/base` 三档变体）；具体实现等级以保真度审计为准。

---

## CLI 快速上手

```bash
# 列出全部 210 个 VLM 架构 ID（`vlm:` 前缀）
python scripts/vlm_zoo.py --list

# 模糊搜索
python scripts/vlm_zoo.py --search llava

# 按年份查看演进时间线
python scripts/vlm_zoo.py --timeline

# Smoke Test（前向推理验证）
python scripts/vlm_zoo.py --smoke vlm:clip_tiny
```

---

## 核心 20 算法族（2021-2023 演进主线）

| # | 算法族 | 年份 | 核心创新 |
|:--|:------|:-----|:---------|
| 1 | CLIP | 2021 | 对比学习对齐 Image-Text，零样本迁移能力开创性突破 |
| 2 | ALIGN | 2021 | 大规模噪声 Image-Text 对训练，Dual Encoder 简洁架构 |
| 3 | ViLT | 2021 | 去除 Region Feature / CNN，纯 Transformer 处理视觉-语言 |
| 4 | SimVLM | 2021 | 简化 VLM 预训练，前缀语言模型 (PrefixLM) 统一目标 |
| 5 | ALBEF | 2021 | Align Before Fuse --- 先对齐再融合，动量蒸馏去噪 |
| 6 | LiT | 2022 | Locked-image Tuning --- 冻结预训练视觉编码器，仅训练文本侧 |
| 7 | BLIP | 2022 | Bootstrapping Language-Image Pre-training + CapFilt 噪声过滤 |
| 8 | CoCa | 2022 | Contrastive Captioners --- 对比学习 + 生成式 Caption 联合训练 |
| 9 | OFA | 2022 | 统一 Seq2Seq 框架，多模态多任务一个模型 |
| 10 | Flamingo | 2022 | 少样本多模态学习，Perceiver Resampler + Gated Cross-Attention |
| 11 | PaLI | 2022 | Pathways Language and Image，超大规模多语言多模态模型 |
| 12 | BLIP-2 | 2023 | Q-Former 桥接冻结视觉编码器与冻结 LLM，训练效率飞跃 |
| 13 | InstructBLIP | 2023 | 指令微调 BLIP-2，多任务指令跟随能力 |
| 14 | LLaVA | 2023 | Visual Instruction Tuning --- MLP 投影 + LLM 指令微调 |
| 15 | MiniGPT-4 | 2023 | 一层线性投影对齐视觉编码器与 Vicuna LLM |
| 16 | Kosmos-2 | 2023 | Grounded Multimodal LLM --- 文本生成 + 目标定位联合 |
| 17 | mPLUG-Owl2 | 2023 | Modality-Adaptive Module 实现多模态协作 |
| 18 | CogVLM | 2023 | Visual Expert Module 注入 LLM 每一层，深度视觉融合 |
| 19 | PaLI-X | 2023 | Scaling up PaLI 至 55B，多任务多语言 SOTA |
| 20 | Qwen-VL | 2023 | 高分辨率视觉编码 + 多粒度文本理解，中英双语 |

---

## 2024-2025 扩展算法族

在上述 20 个核心家族之外，Zoo 还收录了 50 个 2024-2025 年方向的家族，覆盖文档理解 / OCR（Kosmos-2.5, DocOwl2, OCRVLM）、图表与科学图像（ChartVLM, Science-VLM）、网页与界面理解（WebVLM）、视频（Video-Qwen-VL）、多模态代理（Agent-VL, Rabbit-VLM）、端侧轻量推理（EdgeVLM, Bunny）以及 InternVL2、XGen-MM、Aria、LLaMA-Vision、SigLIP-VLM、MixVLM 等通用升级路线。

> 完整列表与变体见 `python scripts/vlm_zoo.py --list`，按年份浏览用 `--timeline`。

---

## 演进脉络

```mermaid
graph LR
    A["CLIP / ALIGN<br/>对比学习"] --> B["ALBEF / BLIP<br/>对齐+融合"]
    B --> C["BLIP-2<br/>Q-Former 桥接 LLM"]
    C --> D["InstructBLIP / LLaVA<br/>指令微调"]
    A --> E["ViLT / SimVLM<br/>端到端 Transformer"]
    E --> F["CoCa / OFA<br/>统一框架"]
    F --> D
    C --> G["Flamingo / PaLI<br/>超大规模少样本"]
    G --> H["CogVLM / Qwen-VL<br/>深度融合"]
```

---

## 架构分类

### 对比学习 (Contrastive Learning)

以 Image-Text 对比损失为核心的双塔模型。

| 算法族 | 视觉编码器 | 文本编码器 | 特点 |
|:------|:----------|:----------|:-----|
| CLIP | ViT / ResNet | Transformer | 零样本迁移基线 |
| ALIGN | EfficientNet | BERT | 18 亿噪声数据训练 |
| LiT | ViT (冻结) | Transformer (可训练) | 仅微调文本侧 |

### 对齐 + 融合 (Align & Fuse)

先对齐表示空间，再通过 Cross-Attention 深度融合。

| 算法族 | 核心机制 | 特点 |
|:------|:---------|:-----|
| ALBEF | Momentum Distillation | 动量蒸馏 + ITC/ITM/MLM |
| BLIP | CapFilt | 噪声 Caption 自动过滤 |
| CoCa | Contrastive + Captioning | 双目标联合优化 |

### 桥接 LLM (Bridge to LLM)

将预训练视觉编码器与冻结 LLM 高效连接。

| 算法族 | 桥接方式 | LLM |
|:------|:---------|:----|
| BLIP-2 | Q-Former | OPT / FlanT5 |
| Flamingo | Perceiver Resampler | Chinchilla |
| LLaVA | MLP Projection | LLaMA / Vicuna |
| MiniGPT-4 | Linear Projection | Vicuna |

### 指令微调 (Instruction Tuning)

通过多任务指令数据增强模型的跟随能力。

| 算法族 | 基座 | 关键数据 |
|:------|:-----|:---------|
| InstructBLIP | BLIP-2 | 多任务指令数据集 |
| LLaVA | LLaMA | GPT-4 生成的视觉指令数据 |
| Qwen-VL | Qwen | 多粒度中英指令数据 |

---

## 用法示例

```bash
# 构建并前向验证任意家族的任意变体
python scripts/vlm_zoo.py --smoke vlm:clip_tiny
python scripts/vlm_zoo.py --smoke vlm:agent_vl_tiny
```
