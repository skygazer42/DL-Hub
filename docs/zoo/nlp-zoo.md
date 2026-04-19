---
icon: material/text
---

# NLP Zoo

> **49 算法族 / 814 Architecture IDs** --- 覆盖 Transformer、RNN、CNN、MLP 等全部主流文本编码器架构。

---

## CLI 快速上手

```bash
# 列出全部 814 个架构 ID
python -m zoo.nlp --list

# 模糊搜索
python -m zoo.nlp --search bert

# Smoke Test（前向推理验证）
python -m zoo.nlp --smoke bert_base
```

---

## 架构分类总览

| 类别 | 代表算法 | 说明 |
|:-----|:---------|:----|
| Transformer | BERT, GPT, T5, ALBERT, DistilBERT, Longformer, BigBird | 基于 Self-Attention 的主流文本编码器 |
| 高效 Transformer | Performer, Nystromformer, FNet, Synthesizer, Linformer | 降低 Attention 复杂度至线性或亚二次 |
| RNN 系列 | LSTM, GRU, BiLSTM, BiGRU, IndRNN, SRU, QRNN | 循环神经网络及其变体 |
| CNN 系列 | TextCNN, InceptionCNN, DPCNN, VDCNN, ResConv | 卷积文本编码器 |
| MLP 系列 | gMLP, ResMLP, MLP-Mixer | 纯 MLP 文本编码 |
| 轻量级 | FastText, WaveNet, TCN | 面向高吞吐场景的轻量模型 |

---

## Transformer 编码器

Transformer 架构自 2017 年提出以来已成为 NLP 领域的绝对主流。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| BERT | bert_tiny/mini/small/base/large | Masked Language Model + Next Sentence Prediction |
| GPT | gpt_small/medium/large/xl | 自回归语言模型，单向 Attention |
| T5 | t5_small/base/large/3b/11b | Text-to-Text 统一框架 |
| ALBERT | albert_base/large/xlarge/xxlarge | 参数共享 + Factorized Embedding |
| DistilBERT | distilbert_base | 知识蒸馏压缩 BERT |
| Longformer | longformer_base/large | Local + Global Sliding Window Attention |
| BigBird | bigbird_base/large | Sparse Attention = Random + Window + Global |

!!! tip "统一构建接口"

    ```python
    from zoo.nlp import build

    model = build("bert_base", num_classes=2)
    ```

---

## 高效 Transformer

针对标准 Transformer 的 $O(n^2)$ Attention 复杂度进行优化。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| Performer | performer_small/base/large | FAVOR+ 随机特征近似 Softmax Attention |
| Nystromformer | nystromformer_base | Nystrom 方法近似 Attention 矩阵 |
| FNet | fnet_base/large | 用 Fourier Transform 替代 Self-Attention |
| Synthesizer | synthesizer_dense/random/factorized | 合成 Attention 权重，不依赖 Query-Key 点积 |
| Linformer | linformer_base | 低秩线性投影压缩 Key/Value |

!!! note "复杂度对比"

    | 方法 | Attention 复杂度 |
    |:----|:----------------|
    | Standard Transformer | $O(n^2 d)$ |
    | Linformer | $O(n k d)$, $k \ll n$ |
    | Performer | $O(n r d)$, $r \ll n$ |
    | FNet | $O(n \log n)$ |

---

## RNN 系列

循环神经网络在序列建模中仍有广泛应用，尤其在资源受限场景下。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| LSTM | lstm_1L/2L/3L_128/256/512 | 门控机制解决梯度消失 |
| GRU | gru_1L/2L/3L_128/256/512 | 简化门控，参数更少 |
| BiLSTM | bilstm_1L/2L_128/256/512 | 双向上下文编码 |
| BiGRU | bigru_1L/2L_128/256/512 | 双向 GRU |
| IndRNN | indrnn_2L/4L/6L | 独立循环连接，支持更深网络 |
| SRU | sru_2L/4L/8L | Simple Recurrent Unit，高度并行化 |
| QRNN | qrnn_2L/4L | Quasi-RNN，卷积 + 循环门控混合 |

---

## CNN 系列

卷积编码器以固定感受野捕获局部 n-gram 特征，推理速度快。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| TextCNN | textcnn_k3/k4/k5/multi | 多尺度 1D 卷积 + Max Pooling |
| InceptionCNN | inceptioncnn_small/base | Inception-style 多尺度并行卷积 |
| DPCNN | dpcnn_base | Deep Pyramid CNN，固定特征图大小的深层卷积 |
| VDCNN | vdcnn_9/17/29/49 | Very Deep CNN，字符级深层卷积 |
| ResConv | resconv_4L/8L | 残差连接 + 1D 卷积 |

---

## MLP 系列

纯 MLP 架构在 NLP 中的探索。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| gMLP | gmlp_tiny/small/base | Spatial Gating Unit 替代 Attention |
| ResMLP | resmlp_12/24/36 | 残差 MLP + Cross-patch Sublayer |
| MLP-Mixer | mlp_mixer_s16/b16/l16 | Token-Mixing + Channel-Mixing |

---

## 轻量级模型

面向高吞吐、低延迟场景的轻量文本模型。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| FastText | fasttext_base | 浅层模型 + n-gram 特征，极速训练 |
| WaveNet | wavenet_nlp_small/base | 因果扩张卷积 (Dilated Causal Conv) |
| TCN | tcn_4L/8L_64/128/256 | Temporal Convolutional Network，全并行序列建模 |

---

## 用法示例

=== "分类任务"

    ```python
    from zoo.nlp import build

    model = build("bert_base", num_classes=4)
    # model(input_ids, attention_mask) -> logits [B, 4]
    ```

=== "特征提取"

    ```python
    from zoo.nlp import build

    model = build("longformer_base", num_classes=0)  # num_classes=0 返回特征
    # model(input_ids, attention_mask) -> features [B, D]
    ```

=== "CLI Smoke Test"

    ```bash
    # 验证 T5 Small 可正常前向推理
    python -m zoo.nlp --smoke t5_small
    ```
