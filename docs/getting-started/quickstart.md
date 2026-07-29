# 5 分钟体验

跟着下面四步，快速跑通你的第一节 DL-Hub 课程。

---

## Step 1: 克隆并安装

```bash
git clone https://github.com/skygazer42/DL-Hub.git
cd DL-Hub
python -m venv .venv && source .venv/bin/activate
# CPU 示例；CUDA/macOS 命令请按安装指南选择
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[vision]"
```

!!! note "详细安装步骤"

    如需 PyTorch GPU 版本或 conda 安装，请参阅 [安装指南](installation.md)。

---

## Step 2: 运行 Smoke Check

```bash
python scripts/smoke_check.py
```

该命令运行覆盖 8 个 track 的精选离线案例，用于确认 Python、PyTorch 等核心依赖已就绪；
它不是 339 个课程的全量训练。使用 `python scripts/smoke_check.py --list` 可查看覆盖清单。

---

## Step 3: 运行第一节课

使用 `--dataset fake` 跳过数据下载，直接用随机数据跑通训练流程：

```bash
python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake --epochs 1
```

!!! success "预期行为"

    你会看到训练日志输出，包括每个 epoch 的 loss 和准确率。
    使用 fake 数据时，准确率无实际意义，但 **代码能跑通即说明环境正确**。

---

## Step 4: 查看所有课程

列出全部赛道：

```bash
python scripts/run_lesson.py --list
```

列出某条赛道下的所有课程：

```bash
python scripts/run_lesson.py vision --list
python scripts/run_lesson.py nlp --list
python scripts/run_lesson.py gnn --list
```

---

## 统一入口与课程参数

所有训练课程共享 `--seed`、`--device`、`--run-name` 三个核心参数；其他参数根据任务提供：

| 参数 | 适用范围 | 说明 |
|------|----------|------|
| `--seed` | 所有训练课程 | 随机种子，确保可复现 |
| `--device` | 所有训练课程 | `auto` / `cpu` / `cuda` / `mps` |
| `--run-name` | 所有训练课程 | 输出目录中的运行名称 |
| `--dataset` | 部分真实数据课程 | 这些课程支持 `fake` 离线模式 |
| `--epochs` / `--steps` | 按任务提供 | 训练轮数或优化步数 |
| `--batch-size` | 按任务提供 | 每批样本数 |
| `--learning-rate` / `--lr` | 按任务提供 | 学习率 |
| `--max-train-batches` / `--max-eval-batches` | 按任务提供 | 限制批次数，便于快速验证 |

查询任意课程的准确参数：

```bash
python scripts/run_lesson.py gnn lesson_01_compact_graph_classification --describe
```

!!! example "组合使用示例"

    ```bash
    python -m tracks.vision.lesson_01_mnist_lenet.train \
        --dataset fake \
        --epochs 2 \
        --batch-size 32 \
        --device cpu \
        --max-train-batches 5
    ```

---

## 其他赛道快速体验

### GNN 赛道

```bash
python -m tracks.gnn.lesson_01_compact_graph_classification.train --epochs 1
```

### NLP 赛道

```bash
python -m tracks.nlp.lesson_01_compact_text_classification.train --epochs 1
```

### Foundations 赛道

```bash
python -m tracks.foundations.lesson_01_tensors.run
```

---

!!! tip "两种离线数据方式"

    MNIST 等可选真实数据课程使用 `--dataset fake` 生成同形状的测试数据；
    GNN、NLP、LLM 等 synthetic 课程本身就生成内置数据，因此不接受也不需要该参数。

    如果课程提供 `--max-train-batches` / `--max-eval-batches`，可用它们缩短训练与评估；
    具体能力以 `run_lesson.py ... --describe` 输出为准。

---

## 下一步

- :material-map-marker-path: 选择一条 [学习赛道](../tracks/index.md) 开始系统学习
- :material-archive: 浏览 [Model Zoo](../zoo/index.md) 的注册目录与保真度审计
- :material-cog: 阅读 [项目结构](../developer/structure.md)，理解代码组织方式
