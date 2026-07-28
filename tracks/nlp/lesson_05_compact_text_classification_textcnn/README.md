# Lesson 05：TextCNN（compact 文本分类）

这节课在同一个 compact 文本分类任务上，换一种建模方式：**TextCNN**。

核心点：

- embedding 后做多组 1D convolution（不同 kernel size）
- 每个卷积通道做 global max pooling
- 拼接后接分类器

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.nlp.lesson_05_compact_text_classification_textcnn.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_05_compact_text_classification_textcnn/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

