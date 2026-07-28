# Lesson 09：Transformer Summarization（compact encoder-decoder）

这节课实现一个最小的 **Encoder-Decoder Transformer** 摘要模型：

- Encoder：对 source token 序列做自注意力编码
- Decoder：带 **causal mask** 的自回归解码
- 训练：使用 **teacher forcing**，按 token 计算交叉熵、token accuracy、exact match

数据是合成的：输入是一串 token，目标摘要是从 source 中抽取少量“关键信息”位置构成的短序列，不需要下载任何数据。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.nlp.lesson_09_compact_transformer_summarization.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_09_compact_transformer_summarization/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
