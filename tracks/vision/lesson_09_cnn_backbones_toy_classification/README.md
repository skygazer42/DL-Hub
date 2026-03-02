# Lesson 09 — CNN Backbones (VGG / ResNet / DenseNet / RepVGG / MobileNet / ShuffleNet / EfficientNet)

目标：把常见 CNN 主干网络用一致风格实现（含 VGG / ResNet / DenseNet / RepVGG / MobileNet / ShuffleNet / EfficientNet 等），并在同一套 toy 数据集上跑通分类闭环。

## 运行

CPU 冒烟（默认 `resnet18`）：

```bash
python -m tracks.vision.lesson_09_cnn_backbones_toy_classification.train \
  --device cpu --epochs 1 \
  --num-samples 256 --batch-size 32 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

选择架构：

```bash
python -m tracks.vision.lesson_09_cnn_backbones_toy_classification.train \
  --arch vgg --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

torchvision 模型 zoo（需要已安装 `torchvision`）：

```bash
python -m tracks.vision.lesson_09_cnn_backbones_toy_classification.train \
  --arch tv:resnet50 --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

`timm` 模型 zoo（可选，安装 `timm` 后可用）：

```bash
python -m tracks.vision.lesson_09_cnn_backbones_toy_classification.train \
  --arch timm:resnet50 --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

列出所有可用架构：

```bash
python -m tracks.vision.lesson_09_cnn_backbones_toy_classification.train --list-arch
```

如果你也想把 `timm` 的模型名一起列出来（列表可能非常大）：

```bash
python -m tracks.vision.lesson_09_cnn_backbones_toy_classification.train --list-arch --include-timm
```

支持（本地实现）：

- `--arch vgg`
- `--arch resnet18` / `--arch resnet34` / `--arch resnet50` / `--arch resnext50`
- `--arch densenet`
- `--arch squeezenet` / `--arch shufflenetv2`
- `--arch mobilenetv1` / `--arch mobilenetv2`
- `--arch efficientnetb0`（或 `efficientnet`）
- `--arch repvgg`（或 `revgg`）

支持（torchvision zoo）：

- `--arch tv:<name>`（示例：`tv:resnet50`、`tv:convnext_tiny`、`tv:vit_b_16`）

支持（torchvision quantized zoo）：

- `--arch tvq:<name>`（示例：`tvq:resnet18`、`tvq:mobilenet_v3_large`）

支持（timm zoo）：

- `--arch timm:<name>`（示例：`timm:resnet50`、`timm:efficientnet_b0`、`timm:vit_base_patch16_224`）

输出目录：

- `outputs/vision/lesson_09_cnn_backbones_toy_classification/<run_name>/config.json`
- `outputs/vision/lesson_09_cnn_backbones_toy_classification/<run_name>/metrics.jsonl`
- `outputs/vision/lesson_09_cnn_backbones_toy_classification/<run_name>/checkpoints/checkpoint.pt`
