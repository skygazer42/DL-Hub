# 运行与实验约定（Conventions）

## 目标

统一入口、核心参数和目录习惯，让不同任务之间的迁移成本保持可控；
任务专属参数不强行做成无意义的占位选项。

课程规模、数据来源和验证强度是三个独立维度；命名规则与完整运行链路见
[实现契约：从课程到可验证系统](implementation-contract.md)。

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

所有 lesson 的 `train.py` 必须支持：

- `--seed`
- `--device`
- `--run-name`

根据任务形态按需支持：

- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--max-train-batches` / `--max-steps`（用于快速自检）

可通过下面的命令查询某一课实际支持的参数：

```bash
python scripts/run_lesson.py vision lesson_01_mnist_lenet --describe
```

## 离线数据约定

每个 lesson 必须提供一种不联网的运行方式，分为两类：

- **显式 fake 模式**：课程同时支持真实数据集时，提供 `--dataset fake`
- **内置数据模式**：synthetic 课程默认生成本地数据，不需要 `--dataset`

不要为了表面统一给内置数据课程增加无效的 `--dataset fake` 参数。

## Checkpoint 安全

`dlhub.checkpoint.load_checkpoint()` 默认使用 PyTorch 的受限 `weights_only` 加载器；
安全加载失败时不会自动降级到 unrestricted pickle。只有文件来源完全可信、且确实需要兼容
旧版 checkpoint 时，才显式传入 `allow_unsafe_legacy=True`。该模式会发出运行时警告，
因为恶意 pickle 可在加载时执行任意代码。下载或来源不明的 checkpoint 禁止使用此开关。

## 验收（Acceptance）

每个 lesson 必须给出至少一种可重复的验收：

- 快速截断模式能够在 CPU 上跑通
- 指标阈值
- 或者 pytest 的练习题验收

仓库级静态契约检查不会导入或训练课程：

```bash
python scripts/lesson_contracts.py --check
```

只运行与当前改动有关的最小必要测试；静态 contract、快速 smoke 和 benchmark 不能互相替代。
