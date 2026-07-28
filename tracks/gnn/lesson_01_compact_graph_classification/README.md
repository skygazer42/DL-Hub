# Lesson 01：Compact 图级分类（Cycle vs Star）

本课目标：用一个 **完全合成** 的 compact 数据集，跑通图级分类闭环（数据 → 模型 → 训练 → 评估 → 记录输出）。

我们用两类固定大小图：

- **Cycle（环）**：所有点度数接近（每个点连接左右两个点）
- **Star（星）**：中心点度数很大，其他点度数很小

用一个最小的 **GCN（Graph Convolutional Network）** 做图级分类。

## 你将学到

- 邻接矩阵 `A`、加自环 `A + I`
- 对称归一化 `D^{-1/2} A D^{-1/2}`
- 节点表示 → 图表示（global mean pooling）
- 为什么图任务常见“全图一个样本”的输入形状

## 运行（离线可跑）

从仓库根目录运行：

```bash
python -m tracks.gnn.lesson_01_compact_graph_classification.train \
  --epochs 3 --device cpu --run-name dev
```

快速冒烟（更快）：

```bash
python -m tracks.gnn.lesson_01_compact_graph_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

输出目录遵循统一约定：

```
outputs/gnn/lesson_01_compact_graph_classification/<run_name>/
  config.json
  metrics.jsonl
  checkpoints/checkpoint.pt
  logs/train.log
```

