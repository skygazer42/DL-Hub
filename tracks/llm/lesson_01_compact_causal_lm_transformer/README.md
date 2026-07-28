# Lesson 01：compact Causal LM（Transformer Decoder）

这节课做一个最小的 causal language model：

- 输入：`input_ids`（固定长度序列）
- 目标：预测下一个 token（next-token prediction）
- 模型：Transformer decoder block（causal self-attention）
- 生成：给一个短 prompt，按自回归方式生成后续 token

数据是合成的：token 会按 `+1 (mod V)` 的规则递增，训练非常快，便于看清楚整个闭环。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.llm.lesson_01_compact_causal_lm_transformer.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/llm/lesson_01_compact_causal_lm_transformer/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`：每个 epoch 的生成样例
- `logs/train.log`
- `checkpoints/checkpoint.pt`

