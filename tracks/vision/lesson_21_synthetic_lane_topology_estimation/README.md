# Lesson 21 — Synthetic Lane Topology Estimation

目标：在合成道路图像上同时学习每条车道的热图和车道之间的邻接关系，构建一个最小可跑的 lane graph 预测闭环。

## 任务定义

- 输入：灰度道路图 `(1, H, W)`，包含 2-4 条弯曲车道线
- 输出：
  - `lane_heatmaps`：每条车道一个热图 `(K, H, W)`
  - `adjacency_logits`：车道之间的连接关系 `(K, K)`
- 监督：
  - `lane_heatmaps`：逐车道高斯中心线
  - `adjacency`：相邻车道连边矩阵
  - `lane_presence`：有效车道掩码
- 损失：`MSE(heatmaps) + BCE(adjacency)`

## 运行

```bash
python -m tracks.vision.lesson_21_synthetic_lane_topology_estimation.train \
  --device cpu --epochs 1 \
  --num-samples 128 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

输出目录：

- `outputs/vision/lesson_21_synthetic_lane_topology_estimation/<run_name>/config.json`
- `outputs/vision/lesson_21_synthetic_lane_topology_estimation/<run_name>/metrics.jsonl`
- `outputs/vision/lesson_21_synthetic_lane_topology_estimation/<run_name>/checkpoints/checkpoint.pt`
