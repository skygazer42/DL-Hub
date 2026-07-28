# 如何运行（DL-Hub）

这个仓库正在重置为“PyTorch 主线 + 多轨课程”。推荐从 `tracks/` 下的 lesson 开始跑通，然后再深入。

## Python 环境建议

- Python：建议 3.10+
- 环境管理：`venv` 或 `conda` 均可

示例（venv）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## 快速冒烟验证（推荐）

### 1) 仓库级冒烟（推荐第一步）

安装开发依赖（包含 pytest/ruff 等工具；PyTorch 需要你自行安装适配版本）：

```bash
python -m pip install -r requirements-dev.txt
```

运行仓库级冒烟脚本：

```bash
python scripts/smoke_check.py
```

> 说明：`smoke_check.py` 会优先跑 NumPy 模块的自检；如果检测到已安装 `torch`（以及 `torchvision`），还会额外跑一两个 PyTorch lesson 的离线冒烟（`fake/compact` 模式不下载数据）。如果你还没装 PyTorch，则会跳过 lesson 检查并给出提示。

或者运行测试（如果你已安装 dev 依赖）：

```bash
pytest -q
```

### 2) 第一个 PyTorch lesson：MNIST + LeNet（无需下载的 fake 模式）

最小化跑通（1 epoch + 限制 batch 数，`fake` 模式不会下载数据）：

```bash
python -m tracks.vision.lesson_01_mnist_lenet.train \
  --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

## 统一入口（可选）

如果你不想记每个 lesson 的模块路径，可以用统一入口来列出并运行：

```bash
python scripts/run_lesson.py --list
python scripts/run_lesson.py vision --list
python scripts/run_lesson.py vision lesson_06_swin_compact_classification -- --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

如果你希望跑真实 MNIST（会下载数据到 `.data/`）：

```bash
python -m tracks.vision.lesson_01_mnist_lenet.train \
  --dataset mnist --epochs 1 --max-train-batches 5 --max-eval-batches 2
```

## Vision 模型 zoo（可选）

如果你希望快速浏览/验证 torchvision 提供的视觉模型（分类/分割/检测/光流/视频），可以用：

```bash
python scripts/vision_zoo.py --list
python scripts/vision_zoo.py --smoke tv:resnet18
```

如果你希望快速浏览/验证仓库内置实现的主干网络（无需下载权重），可以用：

```bash
python scripts/vision_zoo.py --list --search dl:
python scripts/vision_zoo.py --smoke dl:resnet18 --num-classes 10
python scripts/vision_zoo.py --smoke dl:vit_tiny --num-classes 10 --image-size 64
```

### 3) 第一个 GNN lesson：compact 图级分类（无需下载）

这个 lesson 完全使用合成数据集（Cycle vs Star），适合先理解图卷积/图池化的最小闭环：

```bash
python -m tracks.gnn.lesson_01_compact_graph_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

如果你想进一步对照 GIN（Graph Isomorphism Network），可以跑：

```bash
python -m tracks.gnn.lesson_02_gin_compact_graph_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

如果你想进一步对照 GAT（Graph Attention Network），可以跑：

```bash
python -m tracks.gnn.lesson_03_gat_compact_graph_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

如果你希望跑一个经典的节点分类数据集（Cora）上的 GCN（半监督，transductive），可以跑：

```bash
python -m tracks.gnn.lesson_04_cora_node_classification_gcn.train \
  --epochs 1 --device cpu --run-name smoke
```

### 4) 第一个 NLP lesson：compact 文本分类（无需下载）

这个 lesson 使用合成文本数据集，包含最小 tokenizer/vocab、padding/mask、embedding mean pooling 的完整闭环：

```bash
python -m tracks.nlp.lesson_01_compact_text_classification.train \
  --epochs 1 --max-train-batches 2 --max-eval-batches 2 --device cpu --run-name smoke
```

## 常用开发命令（可选）

如果你安装了 `ruff/black/isort` 等工具，可以使用：

```bash
make lint
make test
make format
make smoke
```
