# Lesson 10：图像去噪（Synthetic, toy-first）

目标：把“输入带噪图 → 输出干净图”的回归式训练闭环跑通，并对比经典/深度学习去噪方法：

- **BM3D**：传统方法基线（无需训练，CPU 可跑）
- **DnCNN**：工程上很常用的卷积去噪网络
- **Restormer**：强效果（更吃算力）
- **Noise2Noise**：不需要干净标注（训练时用两份独立噪声图配对）
- **Blind-Spot (Noise2Self/Noise2Void)**：不需要干净标注（训练时随机遮挡像素，只在遮挡位置计算重建损失）
- **NAFNet**：现代高效 restoration 网络（纯卷积/门控，效果强）
- **SwinIR**：窗口注意力 Transformer 去噪（toy 版）
- **RIDNet**：残差注意力 CNN 去噪
- **FFDNet / DRUNet**：噪声强度条件化去噪（sigma map）
- **DDPM U-Net**：扩散模型常用的噪声条件 U-Net（toy 版，这里作为条件化残差去噪网络）
- **MIRNet**：多尺度特征融合的 restoration 网络（toy 版）
- **MPRNet**：多阶段逐步细化的 restoration 网络（toy 版）
- **UFormer**：U 形 Transformer restoration 网络（toy 版）
- **CBDNet**：blind denoising（噪声估计 + 条件化去噪，toy 版）
- **DIDN**：Densely connected Iterative Down-Up Network（blind denoising，toy 版）
- **RCAN**：Residual Channel Attention Network（restoration/denoising，toy 版）
- **BSN (Blind-Spot Network)**：方向性盲点网络结构（更“结构化”的 blind-spot，toy 版）
- **PixelCNN-BSN**：PixelCNN masked-conv + 旋转融合的盲点网络（toy 版）

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

### 2.5) 训练 Blind-Spot（Noise2Self/Noise2Void，完全不需要干净数据）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_tiny \
  --train-mode blindspot \
  --blindspot-prob 0.1 \
  --epochs 5
```

### 2.6) 选择更真实的噪声模型（Poisson / Impulse / Shot+Read）

```bash
# Poisson noise (典型 photon noise)
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch bsn:bsn_tiny \
  --train-mode blindspot \
  --noise-type poisson \
  --poisson-peak 30 \
  --epochs 5
```

也可以试试混合/结构噪声：

```bash
# Gaussian + impulse（更接近真实传感器/压缩噪声的混合）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch pixelcnn_bsn:pixelcnn_bsn_tiny \
  --train-mode blindspot \
  --noise-type gaussian_impulse \
  --noise-std 0.1 \
  --impulse-prob 0.03 \
  --epochs 5

# Stripe / banding（条纹/带状噪声）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_tiny \
  --train-mode supervised \
  --noise-type stripe \
  --stripe-amplitude 0.12 \
  --stripe-period 8 \
  --epochs 5
```

### 3) 只跑 BM3D（无需训练）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch bm3d:bm3d_fast \
  --sigma 0.1
```

### 4) 训练 DDPM U-Net（toy supervised）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch ddpm_unet:ddpm_unet_tiny \
  --train-mode supervised \
  --epochs 5
```

### 5) 训练 MIRNet / MPRNet / UFormer（toy supervised）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch uformer:uformer_tiny \
  --train-mode supervised \
  --epochs 5
```

### 6) 训练 Blind Denoising（CBDNet / DIDN / RCAN）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch cbdnet:cbdnet_tiny \
  --train-mode supervised \
  --epochs 5
```

## 代码位置

- 数据：`tracks/vision/lesson_10_synthetic_denoising/data.py`
- 模型：`tracks/vision/lesson_10_synthetic_denoising/model.py`
- 训练：`tracks/vision/lesson_10_synthetic_denoising/train.py`
- 算法实现：`dlhub/vision/denoising/*.py`
