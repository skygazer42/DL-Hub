# Lesson 07：阅读理解（compact span prediction）

这个 lesson 用一个很小的 compact 数据集跑通「阅读理解」的核心形式：**给定 context + question，预测答案在 context 里的起止位置**。

目标不是追求真实数据集的指标，而是把下面这些组件串起来并跑通闭环：

- 数据组织：`context_ids / question_ids / start_idx / end_idx`
- 编码器：BiLSTM（context 与 question 分开编码）
- 匹配：用 question 向量去打分每个 context 位置
- 输出：start / end 两个 logits，分别做交叉熵

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.nlp.lesson_07_reading_comprehension.train \
  --device cpu --epochs 3 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_07_reading_comprehension/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

