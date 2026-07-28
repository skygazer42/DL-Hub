# Lesson 02：compact Chat SFT

这节课把上一节的 causal LM 改成最小可运行的 chat-style supervised fine-tuning：

- 输入：`system/user/assistant` 角色 token 加上少量合成内容 token
- 目标：只在 assistant 回复区间计算 next-token loss
- 数据：一轮合成对话，assistant 学会按固定规则续写 3 个 token，再输出 `eos`
- 生成：给定一个短 prompt，自回归生成 assistant 回复

它依然保持 compact-first、纯 PyTorch、CPU 可冒烟，重点是看清楚 chat SFT 的标签 mask 是怎么构造的。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.llm.lesson_02_compact_chat_sft.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## 输出产物（统一规范）

`outputs/llm/lesson_02_compact_chat_sft/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`：每个 epoch 的生成样例
- `logs/train.log`
- `checkpoints/checkpoint.pt`
