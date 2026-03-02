# Lesson 06：GraphSAGE（Cora，全图训练最小实现）

这节课的目标：在 Cora 上跑通 **GraphSAGE** 的最小实现（纯 PyTorch sparse），理解它和 GCN 的核心差异：

- **GCN**：常用对称归一化 `D^{-1/2}AD^{-1/2}`，更像“平滑 + 线性变换”的组合。
- **GraphSAGE**：强调 **邻居聚合器（mean/sum/pool）** 与可扩展的采样训练（本课先做全图训练，先把直觉跑通）。

> 注意：为了把“概念 → 能跑 → 能改”变得简单，本课使用 **全图训练**（full-batch）。学会之后再扩展到 neighbor sampling。

## 运行方式

```bash
python -m tracks.gnn.lesson_06_graphsage_cora.train --epochs 200 --hidden-features 64
```

快速冒烟：

```bash
python -m tracks.gnn.lesson_06_graphsage_cora.train --epochs 1 --device cpu --run-name smoke
```

## 输出产物（统一规范）

`outputs/gnn/lesson_06_graphsage_cora/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## 练习（建议）

1. 把聚合器从 mean 改成 sum（提示：用未归一化邻接 + 度数归一化）。
2. 加一层（3-layer GraphSAGE），观察过平滑（oversmoothing）是否出现。
3. 在 `dropout` 与 `weight_decay` 上做网格搜索，记录最优验证集指标。
