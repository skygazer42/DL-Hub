# Lesson 04：Seq2Seq + Attention（toy 文本生成）

这节课实现一个最小的 **Encoder-Decoder** 文本生成模型，并加上 **Bahdanau Attention**：

- Encoder：GRU 编码输入序列
- Attention：根据 decoder hidden 对 encoder outputs 做加权汇聚
- Decoder：GRUCell 逐 token 生成输出

数据是合成的：输入是一串 token，输出是**反转后的序列**，不需要下载任何数据。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.nlp.lesson_04_toy_seq2seq_attention_generation.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/nlp/lesson_04_toy_seq2seq_attention_generation/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`：每个 epoch 的少量生成样例
- `logs/train.log`
- `checkpoints/checkpoint.pt`

