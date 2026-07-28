# Lesson 03：Compact 图级分类（GAT）

本课目标：在同一个 compact 图任务（Cycle vs Star）上，实现 **GAT（Graph Attention Network）** 的最小可读版本，并理解“注意力”在图邻居聚合中的作用。

本仓库里的旧实现参考来源：`graph/GAT/`（基于 Cora 的 node classification 工程）。本课先用**图级分类**的 compact 任务，把核心公式跑通；后续会再补充 “node classification + Cora” 的最小实现。

## 你将学到

- 注意力打分：`e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])`
- 邻居 softmax：对每个节点 `i` 的邻居 `j` 做归一化
- 多头注意力：concat / average 的差异
- 图级池化：把节点表示聚合成图表示

## 运行（离线可跑）

从仓库根目录运行：

```bash
python -m tracks.gnn.lesson_03_gat_compact_graph_classification.train \
  --epochs 3 --device cpu --run-name dev
```

快速冒烟：

```bash
python -m tracks.gnn.lesson_03_gat_compact_graph_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

输出目录遵循统一约定（同 `docs/CONVENTIONS.md`）。
