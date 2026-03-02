# Lesson 02：Toy 图级分类（GIN）

本课目标：在 Lesson 01 的 toy 图任务基础上，实现 **GIN（Graph Isomorphism Network）** 的核心更新公式，并和 GCN 做一个“结构上”的对照。

本仓库里曾有一个基于 DGL 的 GIN 工程实现（已迁移并移除；仍可在 Git 历史中追溯）。本课把它重写成 **纯 PyTorch + 可读** 的最小实现，避免引入 DGL 依赖。

## 你将学到

- GIN 的核心形式：
  - `h_v^{k+1} = MLP((1 + eps) * h_v^k + AGG({h_u^k, u in N(v)}))`
- `learn_eps` 的意义（是否区分“自己”和“邻居”）
- 邻居聚合（sum / mean / max）的差异
- 图级池化（sum / mean / max）的差异

## 运行（离线可跑）

从仓库根目录运行：

```bash
python -m tracks.gnn.lesson_02_gin_toy_graph_classification.train \
  --epochs 3 --device cpu --run-name dev
```

快速冒烟：

```bash
python -m tracks.gnn.lesson_02_gin_toy_graph_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

输出目录遵循统一约定（同 `docs/CONVENTIONS.md`）。
