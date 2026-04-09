---
icon: material/image-text
---

# VLM Zoo

> **20 算法族** --- 覆盖 2021-2023 年视觉-语言多模态模型的核心演进路线，从对比学习到指令微调。

---

## CLI 快速上手

```bash
# 列出全部 VLM 架构 ID
python -m zoo.vlm --list

# 模糊搜索
python -m zoo.vlm --search blip

# Smoke Test（前向推理验证）
python -m zoo.vlm --smoke clip_base
```

---

## 全部 20 算法族

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

```python
from zoo.vlm import build

model = build("clip_base")
# model.encode_image(images)  -> image_features  [B, D]
# model.encode_text(texts)    -> text_features   [B, D]
```
