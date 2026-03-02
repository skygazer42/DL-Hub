# Lesson 08：LINE（Karate，节点表示学习）

这节课的目标：在 Karate Club 图上跑通 **LINE 风格**的节点表示学习：

- 用负采样（negative sampling）优化目标
- 支持 1st-order（节点-节点）和 2nd-order（节点-上下文）两种形式
- 得到每个节点的 embedding，并保存到统一输出目录

与 Lesson 07（自编码器）相比，LINE 更像是 “skip-gram on edges”。

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.gnn.lesson_08_line_karate_embedding.train --epochs 1 --steps-per-epoch 20 --batch-size 64 --device cpu --run-name smoke
```

## 输出产物（统一规范）

`outputs/gnn/lesson_08_line_karate_embedding/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `embeddings.pt`：`(N, D)` 的节点 embedding（1st-order 保存 node embedding；2nd-order 保存 node + context）
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 比较 `--order 1` 与 `--order 2` 的效果差异。
2. 把负采样分布从 `deg^0.75` 换成 uniform，观察训练稳定性。
3. 把 embedding 维度从 8/16/64 扫一遍，看看是否更容易过拟合。

