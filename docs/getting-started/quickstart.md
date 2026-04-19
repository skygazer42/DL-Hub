# 5 分钟体验

跟着下面四步，快速跑通你的第一节 DL-Hub 课程。

---

## Step 1: 克隆并安装

```bash
git clone https://github.com/skygazer42/DL-Hub.git
cd DL-Hub
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

!!! note "详细安装步骤"

    如需 PyTorch GPU 版本或 conda 安装，请参阅 [安装指南](installation.md)。

---

## Step 2: 运行 Smoke Check

```bash
python scripts/smoke_check.py
```

确认 Python、PyTorch 等核心依赖均已就绪。

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

## 统一 CLI 参数

所有课程共享以下标准命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | `str` | 赛道默认 | 数据集名称，所有课程均支持 `fake` |
| `--epochs` | `int` | `1` | 训练轮数 |
| `--batch-size` | `int` | `64` | 每批样本数 |
| `--learning-rate` | `float` | `1e-3` | 学习率 |
| `--seed` | `int` | `42` | 随机种子，确保可复现 |
| `--device` | `str` | `auto` | 设备选择：`auto` / `cpu` / `cuda` / `mps` |
| `--max-train-batches` | `int` | `None` | 限制训练批次数（用于快速测试） |
| `--max-eval-batches` | `int` | `None` | 限制评估批次数（用于快速测试） |

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
python -m tracks.gnn.lesson_01_gcn_node_classification.train --dataset fake --epochs 1
```

### NLP 赛道

```bash
python -m tracks.nlp.lesson_01_text_classification_rnn.train --dataset fake --epochs 1
```

### Foundations 赛道

```bash
python -m tracks.foundations.lesson_01_tensor_basics.run
```

---

!!! tip "离线测试利器：`--dataset fake`"

    `--dataset fake` 是 DL-Hub 的核心设计之一。它会生成与真实数据格式相同的随机数据，
    让你在 **无网络、无 GPU** 的环境下也能验证代码逻辑。CI 测试中也大量使用此参数。

    配合 `--max-train-batches 2 --max-eval-batches 2`，可以在几秒内完成一次完整的
    训练-评估循环，非常适合快速迭代和调试。

---

## 下一步

- :material-map-marker-path: 选择一条 [学习赛道](../tracks/index.md) 开始系统学习
- :material-archive: 浏览 [Model Zoo](../zoo/index.md)，了解 8000+ 可用架构
- :material-cog: 阅读 [项目结构](../developer/structure.md)，理解代码组织方式
