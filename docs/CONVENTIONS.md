# 运行与实验约定（Conventions）

## 目标

同一套命令与目录习惯能让你在不同任务间迁移成本极低。

## Seed / 可复现

- 每个训练入口必须支持 `--seed`
- 必须在构建数据集、模型之前设置 seed
- 记录关键版本信息（Python/torch/torchvision）

## 设备选择

- 支持 `--device cpu|cuda|mps|auto`
- 默认 `auto`

## 输出目录结构

统一输出到：

```
outputs/<track>/<lesson>/<run_name>/
  config.json
  metrics.jsonl
  checkpoints/
  plots/ (可选)
```

## 训练命令行参数约定

所有 lesson 的 `train.py` 尽量支持：

- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--seed`
- `--device`
- `--max-train-batches` / `--max-steps`（用于快速自检）

## 验收（Acceptance）

每个 lesson 必须给出至少一种可重复的验收：

- `--max-steps` 模式 10 秒内跑通
- 指标阈值
- 或者 pytest 的练习题验收

