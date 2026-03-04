# Lesson 10：图像去噪（Synthetic, toy-first）

目标：把“输入带噪图 → 输出干净图”的回归式训练闭环跑通，并对比经典/深度学习去噪方法：

- **BM3D**：传统方法基线（无需训练，CPU 可跑）
- **DnCNN**：工程上很常用的卷积去噪网络
- **Restormer**：强效果（更吃算力）
- **Noise2Noise**：不需要干净标注（训练时用两份独立噪声图配对）

## 快速开始

列出所有可用模型：

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train --list-arch
```

### 1) 训练 DnCNN（supervised）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_9 \
  --train-mode supervised \
  --epochs 5
```

### 2) 训练 Noise2Noise（无干净数据的训练范式）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch noise2noise_unet:n2n_unet_tiny \
  --train-mode noise2noise \
  --epochs 5
```

### 3) 只跑 BM3D（无需训练）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch bm3d:bm3d_fast \
  --sigma 0.1
```

## 代码位置

- 数据：`tracks/vision/lesson_10_synthetic_denoising/data.py`
- 模型：`tracks/vision/lesson_10_synthetic_denoising/model.py`
- 训练：`tracks/vision/lesson_10_synthetic_denoising/train.py`
- 算法实现：`dlhub/vision/denoising/*.py`

