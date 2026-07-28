# Lesson 10：Prompt Tuning Classifier（compact 文本分类）

这节课把前面的 Transformer 文本分类器改造成一个更贴近参数高效微调的版本：

- 冻结 token embedding、位置编码和 Transformer 编码器
- 只学习一小段可训练的 soft prompt
- 保留一个轻量分类头，把 pooled 表示映射成二分类 logits

这能帮你看清楚：当主干网络不更新时，prompt token 仍然可以改变上下文表示并完成任务适配。

## 运行方式

```bash
python -m tracks.nlp.lesson_10_compact_prompt_tuning_classifier.train \
  --epochs 3 \
  --prompt-length 4 \
  --embed-dim 64 \
  --num-heads 4 \
  --num-layers 2
```

快速冒烟：

```bash
python -m tracks.nlp.lesson_10_compact_prompt_tuning_classifier.train \
  --epochs 1 \
  --prompt-length 4 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu \
  --run-name smoke
```

## 输出产物

`outputs/nlp/lesson_10_compact_prompt_tuning_classifier/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 你会学到什么

1. soft prompt 如何在不改动 backbone 权重的情况下插入到序列前缀
2. 冻结 backbone 时，哪些参数还需要继续训练
3. 参数高效微调为什么适合做快速任务适配

## 建议练习

1. 把 pooled 表示改成只读取 prompt token 的平均值，比较效果。
2. 只训练 soft prompt，不训练分类头，观察收敛速度变化。
3. 扩大 `prompt_length`，记录参数量与准确率的关系。
