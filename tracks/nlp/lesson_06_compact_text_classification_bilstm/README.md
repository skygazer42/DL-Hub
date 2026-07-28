# Lesson 06：BiLSTM（compact 文本分类）

这节课继续在同一个 compact 文本分类任务上对比 **RNN 家族**：

- embedding 后用 BiLSTM 编码序列
- 取最终 hidden（forward + backward）作为句向量
- 送入分类器

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.nlp.lesson_06_compact_text_classification_bilstm.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_06_compact_text_classification_bilstm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

