# Lesson 03：NER（compact，BiLSTM 序列标注）

这节课的目标：从文本分类推进到 **序列标注**（token-level 分类），跑通一个最小 NER 闭环：

- 输入：`input_ids (B, T)` + `attention_mask (B, T)`
- 输出：`logits (B, T, num_tags)`
- loss：对每个 token 做交叉熵（pad 位置用 `ignore_index` 跳过）

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.nlp.lesson_03_compact_ner_bilstm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_03_compact_ner_bilstm/<run_name>/`

- `config.json`
- `vocab.json`
- `tags.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 把 BiLSTM 换成 Transformer encoder（复用 Lesson 02 的 attention 结构）。
2. 增加实体类型或引入多 token 实体，让任务更接近真实 NER。
3. 增加一个简单的 CRF 层，对比 token-level CE 与 CRF 的差异。

