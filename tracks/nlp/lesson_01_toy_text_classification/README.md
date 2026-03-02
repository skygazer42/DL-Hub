# Lesson 01：Toy 文本分类（Embedding + Mean Pooling）

本课目标：用一个 **完全合成** 的文本数据集，跑通 NLP 的最小闭环：

- 文本 → token → id（最小 tokenizer/vocab）
- padding/attention_mask
- embedding + mean pooling → 分类
- 训练/评估/输出记录（统一输出目录结构）

> 为什么是 toy？因为这样你可以先把“数据管线与模型形状”弄明白，再把同样结构迁移到 IMDb/AGNews 等真实数据集。

## 运行（离线可跑）

从仓库根目录运行：

```bash
python -m tracks.nlp.lesson_01_toy_text_classification.train \
  --epochs 3 --device cpu --run-name dev
```

快速冒烟：

```bash
python -m tracks.nlp.lesson_01_toy_text_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

输出目录：

```
outputs/nlp/lesson_01_toy_text_classification/<run_name>/
  config.json
  metrics.jsonl
  vocab.json
  checkpoints/checkpoint.pt
  logs/train.log
```

