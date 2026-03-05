# Lesson 10：图像去噪（Synthetic, toy-first）

目标：把”输入带噪图 → 输出干净图”的回归式训练闭环跑通，并对比 **32 种**经典/深度学习去噪方法，覆盖 supervised、noise2noise、blind-spot 三种训练范式和 15 种噪声模型。

## 运行

CPU 冒烟（默认 `dncnn:dncnn_9`，几秒内跑完）：

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --device cpu --epochs 1 \
  --num-samples 256 --batch-size 32 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

列出所有可用模型：

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train --list-arch
```

选择架构（格式 `--arch <family>:<variant>`）：

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch restormer:restormer_tiny --epochs 5
```

## 模型一览

本课包含 32 个去噪算法族，按设计思路分为 5 类：

### 传统方法

| 算法 | 核心思路 | 特点 |
|---|---|---|
| **BM3D** | 块匹配 + 3D 协同滤波 | 无需训练，CPU 直接跑，经典基线 |

### 有监督 CNN / Transformer

| 算法 | 核心思路 | 特点 |
|---|---|---|
| **DnCNN** | 残差学习（学噪声而非干净图） | 工程常用，结构简单，训练快 |
| **RIDNet** | 残差 + 通道注意力 CNN | 特征自适应加权 |
| **NAFNet** | 纯卷积 + 门控（无 attention/norm） | 现代高效 restoration，效果强 |
| **Restormer** | 转置注意力 Transformer | 效果好，算力需求高 |
| **SwinIR** | 窗口注意力 Transformer | Swin Transformer 做 restoration |
| **MIRNet** | 多尺度残差 + 跨尺度特征融合 | 同时利用高/低分辨率信息 |
| **MPRNet** | 多阶段逐步细化 | 每阶段从粗到精恢复 |
| **UFormer** | U 形 Transformer（窗口注意力） | 编码器-解码器 + skip connection |
| **RCAN** | 残差通道注意力网络（深层残差） | 超分辨率/去噪通用 backbone |
| **REDNet** | 对称卷积/反卷积自编码器 + skip | 经典 encoder-decoder restoration |
| **DRRN** | 递归残差单元（权重共享） | 参数少、深度靠 recursion 堆出来 |
| **MemNet** | 持久记忆块（短期+长期融合） | 通过 memory 复用多层特征 |
| **RDN** | 残差密集块（dense 连接） | 多级特征拼接融合，细节恢复强 |
| **PRIDNet** | 金字塔分支 + 注意力 | 多感受野融合，适合纹理/细节 |
| **DHDN** | 普通卷积 + 膨胀卷积混合 | 同时抓局部与更大上下文 |
| **EDSR** | 深残差块（无 BN） | 经典强 CNN backbone（SR/denoise 通用） |
| **ResUNet** | 残差 U-Net | 结构简单稳定，baseline 很好用 |
| **Attention U-Net** | 注意力门控 skip connection | 让 skip 更“选择性”地传递信息 |
| **U-Net++** | 嵌套/密集 skip connection | 更强的多尺度融合（更吃算力） |
| **MWCNN** | Haar wavelet 多尺度特征 | wavelet 下采样/上采样，结构高效 |
| **HINet** | Half InstanceNorm 残差块 | 纹理/细节恢复常用技巧（toy 版） |

### 噪声条件化

| 算法 | 核心思路 | 特点 |
|---|---|---|
| **FFDNet** | 拼接 sigma map 作为额外输入通道 | 一个模型适配多种噪声强度 |
| **DRUNet** | U-Net + sigma map 条件化 | 比 FFDNet 更强，常做即插即用先验 |
| **DDPM U-Net** | 扩散模型的噪声条件 U-Net | 时间步嵌入 + 残差块 |

### Blind Denoising（噪声未知）

| 算法 | 核心思路 | 特点 |
|---|---|---|
| **CBDNet** | 噪声估计子网 + 条件化去噪子网 | 先估噪声 level，再去噪 |
| **DIDN** | 密集连接 + 迭代下采样-上采样 | 多尺度特征复用 |

### 无需干净数据

| 算法 | 核心思路 | 特点 |
|---|---|---|
| **Noise2Noise** | 两张独立噪声图配对训练 | 不需要干净标注，只需两次采样 |
| **BSN** | 方向性盲点卷积 | 结构保证感受野不含中心像素 |
| **PixelCNN-BSN** | PixelCNN masked-conv + 旋转融合 | 自回归风格的盲点 |
| **Gated PixelCNN-BSN** | 带门控的 PixelCNN masked-conv | 门控提升表达力 |
| **DBSN** | 多膨胀率方向盲点特征融合 | 更大感受野的盲点网络 |

## 三种训练范式

### 1) Supervised（默认）

输入带噪图，目标是干净图，用 MSE 损失直接监督。

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_9 \
  --train-mode supervised \
  --epochs 5
```

### 2) Noise2Noise

训练时用**两份独立噪声图配对**（同一张干净图加两次独立噪声），不需要干净标注。原理：两份独立噪声的期望相同，MSE 损失的最优解仍是干净图。

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch noise2noise_unet:n2n_unet_tiny \
  --train-mode noise2noise \
  --epochs 5
```

### 3) Blind-Spot（Noise2Self / Noise2Void）

完全不需要干净数据。随机遮挡部分像素，用邻居像素替代，只在被遮挡位置计算重建损失（`MaskedMSELoss`）。模型无法”作弊”直接拷贝输入，只能学会去噪。

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_tiny \
  --train-mode blindspot \
  --blindspot-prob 0.1 \
  --epochs 5
```

BSN/DBSN/PixelCNN-BSN 等网络从**结构上**保证盲点性质，可以搭配 `--train-mode blindspot` 或 `--train-mode supervised` 使用。

## 噪声模型

本课支持 15 种噪声模型，通过 `--noise-type` 切换：

| 噪声类型 | 说明 | 关键参数 |
|---|---|---|
| `gaussian`（默认） | 加性高斯白噪声 | `--noise-std 0.1` |
| `gaussian_var` | 高斯噪声，sigma 在范围内随机 | `--noise-std-min 0.05 --noise-std-max 0.2` |
| `gaussian_impulse` | 高斯 + 椒盐 | `--noise-std 0.1 --impulse-prob 0.03` |
| `poisson` | 泊松噪声（光子噪声） | `--poisson-peak 30` |
| `poisson_gaussian` | 泊松 + 读出高斯（Poisson-Gaussian） | `--poisson-peak 30 --read-noise 0.02` |
| `impulse` | 纯椒盐噪声 | `--impulse-prob 0.03` |
| `shot_read` | 散粒噪声 + 读出噪声（异方差） | `--shot-noise 0.2 --read-noise 0.02` |
| `speckle` | 乘性散斑噪声 | `--speckle-std 0.15` |
| `speckle_read` | 散斑 + 读出噪声 | `--speckle-std 0.15 --read-noise 0.02` |
| `stripe` | 条纹/带状噪声 | `--stripe-amplitude 0.12 --stripe-period 8 --stripe-direction vertical` |
| `correlated_gaussian` | 空间相关高斯噪声（模糊相关） | `--noise-std 0.1` |
| `quantization` | 量化噪声（ADC / bit-depth） | `--quant-bits 8 --no-quant-dither` |
| `dead_hot` | 传感器坏点（dead/hot pixels） | `--defect-prob 0.002 --defect-hot-ratio 0.5` |
| `rowcol_bias` | 行/列随机偏置（非周期 banding / FPN） | `--row-bias-std 0.02 --col-bias-std 0.02` |
| `mixed` | 混合噪声（shot+read → impulse → quant） | `--shot-noise 0.2 --read-noise 0.02 --impulse-prob 0.03 --quant-bits 8` |

示例：

```bash
# Poisson noise（典型 photon noise）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch bsn:bsn_tiny \
  --train-mode blindspot \
  --noise-type poisson \
  --poisson-peak 30 \
  --epochs 5

# Gaussian sigma range（同一个 batch 内每张图噪声强度不同）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dbsn:dbsn_tiny \
  --train-mode blindspot \
  --noise-type gaussian_var \
  --noise-std-min 0.05 \
  --noise-std-max 0.2 \
  --epochs 5

# Gaussian + impulse（更接近真实传感器/压缩噪声的混合）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch pixelcnn_bsn:pixelcnn_bsn_tiny \
  --train-mode blindspot \
  --noise-type gaussian_impulse \
  --noise-std 0.1 \
  --impulse-prob 0.03 \
  --epochs 5

# Speckle + read（乘性 speckle + 加性读出噪声）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch gated_pixelcnn_bsn:gated_pixelcnn_bsn_tiny \
  --train-mode blindspot \
  --noise-type speckle_read \
  --speckle-std 0.15 \
  --read-noise 0.02 \
  --epochs 5

# Stripe / banding（条纹/带状噪声）
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_tiny \
  --train-mode supervised \
  --noise-type stripe \
  --stripe-amplitude 0.12 \
  --stripe-period 8 \
  --stripe-direction random \
  --epochs 5
```

## 更多示例

### BM3D 基线（无需训练）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch bm3d:bm3d_fast \
  --sigma 0.1
```

### DDPM U-Net（噪声条件化）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch ddpm_unet:ddpm_unet_tiny \
  --train-mode supervised \
  --epochs 5
```

### MIRNet / MPRNet / UFormer

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch mirnet:mirnet_tiny --train-mode supervised --epochs 5
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch mprnet:mprnet_tiny --train-mode supervised --epochs 5
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch uformer:uformer_tiny --train-mode supervised --epochs 5
```

### CBDNet / DIDN / RCAN（blind denoising）

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch cbdnet:cbdnet_tiny --train-mode supervised --epochs 5
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch didn:didn_tiny --train-mode supervised --epochs 5
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch rcan:rcan_tiny --train-mode supervised --epochs 5
```

## 评估指标

- **MSE**（Mean Squared Error）：预测图与干净图的像素级均方误差，越小越好
- **PSNR**（Peak Signal-to-Noise Ratio）：`10 * log10(1 / MSE)`，单位 dB，越高越好；toy 数据上通常在 20–40 dB 范围

训练过程中每个 epoch 输出 `train_mse / eval_mse / eval_psnr`，记录在 `metrics.jsonl` 中。

## 输出产物（统一规范）

`outputs/vision/lesson_10_synthetic_denoising/<run_name>/`

- `config.json` — 完整训练/数据配置
- `metrics.jsonl` — 逐 epoch 指标（train_mse, eval_mse, eval_psnr, lr）
- `logs/train.log` — 训练日志
- `checkpoints/checkpoint.pt` — 模型权重

## 练习（建议）

1. **对比基线**：先跑 `bm3d:bm3d_fast`，记下 PSNR，再用 `dncnn:dncnn_9` supervised 训练 5 epoch，对比深度学习 vs 传统方法
2. **训练范式对比**：固定 `dncnn:dncnn_tiny`，分别用 `--train-mode supervised / noise2noise / blindspot` 训练，对比三种范式在相同 epoch 下的 PSNR
3. **噪声鲁棒性**：用 `gaussian_var`（随机 sigma）训练一个模型，然后用固定 sigma 测试，看模型能否泛化到不同噪声强度
4. **条件化 vs 非条件化**：对比 `dncnn`（不感知 sigma）和 `ffdnet`（拼接 sigma map）在 `gaussian_var` 噪声下的表现
5. **结构盲点 vs 训练盲点**：对比 `bsn:bsn_tiny --train-mode supervised` 和 `dncnn:dncnn_tiny --train-mode blindspot`，理解”结构保证”和”训练技巧”两条路的区别
6. **噪声模型探索**：用 `stripe` 噪声训练，观察方向性噪声对不同模型的影响

## 代码位置

- 数据：`tracks/vision/lesson_10_synthetic_denoising/data.py`
- 模型分发：`tracks/vision/lesson_10_synthetic_denoising/model.py`
- 训练：`tracks/vision/lesson_10_synthetic_denoising/train.py`
- 损失函数：`tracks/vision/lesson_10_synthetic_denoising/losses.py`
- 算法实现：`dlhub/vision/denoising/*.py`（每个算法族一个文件）
