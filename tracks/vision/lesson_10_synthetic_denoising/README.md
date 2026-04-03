# Lesson 10：图像去噪（Synthetic, toy-first）

目标：把"输入带噪图 → 输出干净图"的回归式训练闭环跑通，并对比 **64 种**经典/深度学习去噪方法，覆盖 supervised、noise2noise、blind-spot 三种训练范式和 20 种噪声模型。

## 运行

CPU 冒烟（默认 `dncnn:dncnn_9`，几秒内跑完）：

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --device cpu --epochs 1 \
  --num-samples 256 --batch-size 32 \
  --max-train-batches 2 --max-eval-batches 2 \
  --run-name smoke
```

选择架构（格式 `--arch <family>:<variant>`）：

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch restormer:restormer_tiny --epochs 5
```

### 发现命令

| 用途 | 命令 |
|---|---|
| 列出所有可用模型 | `--list-arch` |
| 按模型族过滤 | `--list-arch --arch-family dncnn` |
| 按关键字过滤 | `--list-arch --arch-match tiny` |
| 只列出模型族名 | `--list-arch-families` |
| 列出所有噪声类型 | `--list-noise-types` |
| 按字母排序 | 追加 `--list-sort alpha` |
| 限制输出行数 | 追加 `--list-limit 20` |
| 打印完整配置 | `--print-config` |

### 测试

```bash
python -m pytest tests/test_tracks_vision_denoising.py -x -q
```

覆盖：20 种噪声模型 DataLoader、46 种深度学习架构前向/反向、17 种传统方法前向、Noise2Noise 配对、Blind-Spot MaskedMSE、传感器缺陷校正效果、CLI 发现命令。

## 数据说明

本课使用**合成数据**（`ToyDenoisingSquares`），无需下载任何真实数据集：

- **干净图**：64×64 灰度图，黑色背景上一个随机位置/大小的白色矩形（像素值 0 或 1）
- **噪声图**：在干净图上叠加噪声，`clamp` 到 `[0, 1]`
- **数据量**：默认 2048 张，80% 训练 / 20% 验证
- **通道**：默认单通道（`--in-channels 1`），可设为 3 测试 RGB 相关噪声

数据简单是刻意设计：让模型在几秒到几分钟内收敛，聚焦于理解算法结构和训练范式的差异，而非炼丹调参。

## 核心概念

**残差学习**：DnCNN 等网络学的不是干净图本身，而是噪声残差 `noise = noisy - clean`。网络输出残差后做减法得到干净图，收敛更快。

**训练范式的理论基础**：

- Supervised：有 `(noisy, clean)` 配对，MSE 损失的最优解就是干净图
- Noise2Noise：有 `(noisy₁, noisy₂)` 配对（同一干净图的两次独立噪声），MSE 最优解的期望仍是干净图（因为 `E[noisy] = clean`）
- Blind-Spot：只有单张噪声图，靠遮住当前像素、用邻居像素预测。只要噪声是像素级独立的，模型只能学到信号而非噪声

**噪声条件化**：FFDNet / DRUNet 等把 sigma map 显式输入网络，让一个模型处理不同噪声强度。相比为每个 sigma 训练一个模型，更灵活更省空间。

**PSNR**：`10 * log10(1 / MSE)`，单位 dB，越高越好。toy 数据上通常 20–40 dB。注意 PSNR 只衡量像素级误差，不直接反映人眼感知质量。

## 方法选择指南

```
你的场景是什么？
│
├─ 有 (noisy, clean) 配对数据
│   ├─ 噪声强度已知且固定 → DnCNN / NAFNet / Restormer
│   ├─ 噪声强度未知或变化 → FFDNet / DRUNet（条件化）
│   ├─ 噪声类型未知       → CBDNet（自动估噪声 level）
│   ├─ 雨线去除           → JORDER / RESCAN / PReNet
│   └─ 想要最强效果       → Restormer / MPRNet / SwinIR
│
├─ 有两份独立噪声观测（无干净图）
│   └─ Noise2Noise
│
├─ 只有单张噪声图（无任何干净数据）
│   ├─ 噪声像素级独立     → Blind-Spot（BSN / DBSN / PixelCNN-BSN）
│   └─ 噪声有空间相关性   → 需要更强假设，先试 BSN 看效果
│
├─ 不想训练 / 只需要基线
│   ├─ 高斯噪声           → BM3D
│   ├─ 椒盐噪声           → Median Filter
│   ├─ Poisson 噪声       → Anscombe + Wiener
│   ├─ 散斑噪声（SAR）    → Lee / Kuan Filter
│   └─ 通用保边去噪       → Bilateral / Non-Local Means
│
└─ 传感器缺陷（非随机噪声）
    ├─ 坏点 (dead/hot)    → Dead/Hot Pixel Corrector
    ├─ 坏行/坏列          → Line Defect Corrector
    ├─ 条纹噪声           → Stripe Remover
    ├─ 行列偏置 (FPN)     → Row/Col Bias Corrector
    └─ 量化色带           → Debanding Filter
```

## 模型一览

本课包含 64 个去噪算法族，按设计思路分为 6 类：

### 传统方法（无需训练）

> 表中 `--arch family` 列为族名，实际运行需写 `--arch <family>:<variant>`（例如 `bm3d:bm3d_fast`），用 `--list-arch --arch-family <family>` 查看可选 variants。

| 算法 | `--arch` family | 核心思路 | 适用噪声 |
|---|---|---|---|
| **BM3D** | `bm3d` | 块匹配 + 3D 协同滤波 | 高斯（经典强基线） |
| **Median Filter** | `median_filter` | 中值滤波 | 椒盐噪声 |
| **Wiener Filter** | `wiener_filter` | 频域最优线性滤波 | 高斯（需功率谱先验） |
| **Guided Filter** | `guided_filter` | 引导滤波（局部线性模型） | 通用保边平滑 |
| **Bilateral Filter** | `bilateral_filter` | 双边滤波（空间+像素值加权） | 通用保边去噪 |
| **Non-Local Means** | `non_local_means` | 非局部均值（块相似度加权） | 高斯（利用自相似性） |
| **Total Variation** | `total_variation` | 全变分正则化（最小化梯度 L1） | 通用（分段常数先验） |
| **Anisotropic Diffusion** | `anisotropic_diffusion` | 各向异性扩散（Perona-Malik） | 通用保边 |
| **Wavelet Shrinkage** | `wavelet_shrinkage` | 小波阈值收缩 | 高斯（多尺度频域） |
| **Anscombe + Wiener** | `anscombe_wiener` | Anscombe 变换 + Wiener 滤波 | Poisson 噪声 |
| **Lee Filter** | `lee_filter` | Lee 滤波（局部统计自适应） | SAR 散斑 |
| **Kuan Filter** | `kuan_filter` | Kuan 滤波（改进 Lee） | 乘性噪声 |

### 有监督 CNN / Transformer

**残差 / 注意力 CNN**

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **DnCNN** | `dncnn` | 残差学习（学噪声而非干净图） | 工程常用，结构简单，训练快 |
| **IRCNN** | `ircnn` | 膨胀卷积残差 CNN | 常用作 PnP/先验 |
| **RIDNet** | `ridnet` | 残差 + 通道注意力 CNN | 特征自适应加权 |
| **DRRN** | `drrn` | 递归残差单元（权重共享） | 参数少、靠 recursion 堆深度 |
| **MemNet** | `memnet` | 持久记忆块（短期+长期融合） | 通过 memory 复用多层特征 |
| **BRDNet** | `brdnet` | 双路径残差网络（BN + dilated） | 两个互补分支融合去噪 |
| **PRIDNet** | `pridnet` | 金字塔分支 + 注意力 | 多感受野融合 |
| **DHDN** | `dhdn` | 普通卷积 + 膨胀卷积混合 | 同时抓局部与更大上下文 |

**密集连接 / 深残差**

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **EDSR** | `edsr` | 深残差块（无 BN） | 经典强 backbone（SR/denoise 通用） |
| **RDN** | `rdn` | 残差密集块（dense 连接） | 多级特征拼接融合 |
| **RRDBNet** | `rrdbnet` | 残差中残差密集块 | ESRGAN backbone，特征复用极深 |
| **CARN** | `carn` | 级联残差网络（group conv） | 轻量高效，参数少 |
| **RCAN** | `rcan` | 残差通道注意力网络 | 超分辨率/去噪通用 |

**U-Net 变体**

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **ResUNet** | `resunet` | 残差 U-Net | 结构简单稳定，baseline 好用 |
| **Attention U-Net** | `attention_unet` | 注意力门控 skip connection | skip 更"选择性"传递信息 |
| **U-Net++** | `unetpp` | 嵌套/密集 skip connection | 更强多尺度融合 |
| **U-Net 3+** | `unet3plus` | 全尺度 skip connection | 比 U-Net++ 更激进的融合 |
| **R2U-Net** | `r2unet` | 循环残差 U-Net | 递归卷积 + 残差连接 |
| **Dense U-Net** | `denseunet` | 密集连接 U-Net | DenseNet 风格编码器-解码器 |
| **REDNet** | `rednet` | 对称卷积/反卷积 + skip | 经典 encoder-decoder |
| **ASPP U-Net** | `aspp_unet` | U-Net + ASPP bottleneck | 多膨胀率上下文聚合 |
| **CBAM U-Net** | `cbam_unet` | U-Net + CBAM 注意力 | 通道+空间特征选择 |
| **ConvNeXt-UNet** | `convnext_unet` | ConvNeXt blocks encoder-decoder | 现代卷积块做 restoration |

**Transformer / 混合结构**

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **Restormer** | `restormer` | 转置注意力 Transformer | 效果好，算力需求高 |
| **SwinIR** | `swinir` | 窗口注意力 Transformer | Swin 做 restoration |
| **UFormer** | `uformer` | U 形 Transformer | 编码器-解码器 + skip |
| **SCUNet** | `scunet` | Conv U-Net + window attention | 混合结构，兼顾局部与全局 |
| **NLRN** | `nlrn` | Non-local + recurrent | 显式引入全局 self-attention |

**多尺度 / 特殊结构**

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **NAFNet** | `nafnet` | 纯卷积 + 门控（无 attn/norm） | 现代高效 restoration |
| **MIRNet** | `mirnet` | 多尺度残差 + 跨尺度融合 | 同时利用高/低分辨率信息 |
| **MPRNet** | `mprnet` | 多阶段逐步细化 | 每阶段从粗到精恢复 |
| **MWCNN** | `mwcnn` | Haar wavelet 多尺度特征 | wavelet 上/下采样，高效 |
| **HINet** | `hinet` | Half InstanceNorm 残差块 | 纹理/细节恢复 |

**去雨 / Deraining（结构化雨线）**

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **JORDER** | `jorder` | 联合雨线检测（mask）+ 去雨（residual） | 显式建模雨线区域，适合 `--noise-type rain` |
| **RESCAN** | `rescan` | 递归/迭代残差去雨 + SE 通道重标定 | 多 stage 逐步细化，偏“迭代优化”风格 |
| **PReNet** | `prenet` | Progressive recurrent（ConvGRU）逐步去雨 | 轻量 recurrent baseline，收敛快 |

### 噪声条件化

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **FFDNet** | `ffdnet` | 拼接 sigma map 作为额外输入通道 | 一个模型适配多种噪声强度 |
| **DRUNet** | `drunet` | U-Net + sigma map 条件化 | 比 FFDNet 更强，常做即插即用先验 |
| **DDPM U-Net** | `ddpm_unet` | 扩散模型的噪声条件 U-Net | 时间步嵌入 + 残差块 |

### Blind Denoising（噪声未知）

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **CBDNet** | `cbdnet` | 噪声估计子网 + 条件化去噪子网 | 先估噪声 level，再去噪 |
| **DIDN** | `didn` | 密集连接 + 迭代下采样-上采样 | 多尺度特征复用 |

### 无需干净数据

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **Noise2Noise** | `noise2noise_unet` | 两张独立噪声图配对训练 | 不需要干净标注，只需两次采样 |
| **BSN** | `bsn` | 方向性盲点卷积 | 结构保证感受野不含中心像素 |
| **PixelCNN-BSN** | `pixelcnn_bsn` | PixelCNN masked-conv + 旋转融合 | 自回归风格的盲点 |
| **Gated PixelCNN-BSN** | `gated_pixelcnn_bsn` | 带门控的 PixelCNN masked-conv | 门控提升表达力 |
| **DBSN** | `dbsn` | 多膨胀率方向盲点特征融合 | 更大感受野的盲点网络 |

### 传感器缺陷校正（无需训练）

| 算法 | `--arch` family | 核心思路 | 特点 |
|---|---|---|---|
| **Stripe Remover** | `stripe_remover` | 条纹/带状噪声去除 | 针对 stripe 型固定模式噪声 |
| **Dead/Hot Pixel Corrector** | `dead_hot_pixel_corrector` | 坏点检测 + 邻域插值 | 修复传感器 dead/hot 像素 |
| **Line Defect Corrector** | `line_defect_corrector` | 坏行/坏列检测 + 修复 | 修复整行/整列的固定缺陷 |
| **Row/Col Bias Corrector** | `rowcol_bias_corrector` | 行列偏置校正（FPN） | 去除固定模式噪声 |
| **Block Bias Corrector** | `block_bias_corrector` | 块级偏置校正 | 去除块状固定模式噪声 |
| **Debanding Filter** | `debanding_filter` | 色带/量化带消除 | 平滑量化阶梯伪影 |

> **Variant 命名规则**：每个算法族通常有 `_tiny`（最小，冒烟用）、`_small`、`_base` 三档。传统方法有 `_fast`、`_quality` 等。
> 完整 variant 列表用 `--list-arch` 查看。
>
> **族名别名**：部分族名支持缩写，如 `unet++`→`unetpp`、`nlm`→`non_local_means`、`tv`→`total_variation`、`ddpm`→`ddpm_unet`、`dead_hot`→`dead_hot_pixel_corrector`。完整别名见 `model.py` 中 `build_model()` 的 `if arch in {...}` 分支。

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

完全不需要干净数据。随机遮挡部分像素，用邻居像素替代，只在被遮挡位置计算重建损失（`MaskedMSELoss`）。模型无法"作弊"直接拷贝输入，只能学会去噪。

```bash
python -m tracks.vision.lesson_10_synthetic_denoising.train \
  --arch dncnn:dncnn_tiny \
  --train-mode blindspot \
  --blindspot-prob 0.1 \
  --epochs 5
```

BSN/DBSN/PixelCNN-BSN 等网络从**结构上**保证盲点性质，可以搭配 `--train-mode blindspot` 或 `--train-mode supervised` 使用。

## 噪声模型

本课支持 20 种噪声模型，通过 `--noise-type` 切换。

### 加性噪声

| 噪声类型 | 说明 | 关键参数 |
|---|---|---|
| `gaussian`（默认） | 加性高斯白噪声 | `--noise-std 0.1` |
| `gaussian_var` | 高斯噪声，sigma 在范围内随机 | `--noise-std-min 0.05 --noise-std-max 0.2` |
| `correlated_gaussian` | 空间相关高斯噪声（模糊相关） | `--noise-std 0.1` |
| `colored_gaussian` | 跨通道相关高斯噪声（RGB 相关） | `--noise-std 0.1 --color-rho 0.5` |
| `poisson` | 泊松噪声（光子噪声） | `--poisson-peak 30` |
| `poisson_gaussian` | 泊松 + 读出高斯（Poisson-Gaussian） | `--poisson-peak 30 --read-noise 0.02` |

### 脉冲 / 乘性噪声

| 噪声类型 | 说明 | 关键参数 |
|---|---|---|
| `impulse` | 纯椒盐噪声 | `--impulse-prob 0.03` |
| `clustered_impulse` | 聚集椒盐噪声（坏点簇） | `--impulse-prob 0.03 --cluster-prob 0.002 --cluster-size 5` |
| `gaussian_impulse` | 高斯 + 椒盐 | `--noise-std 0.1 --impulse-prob 0.03` |
| `speckle` | 乘性散斑噪声 | `--speckle-std 0.15` |
| `speckle_read` | 散斑 + 读出噪声 | `--speckle-std 0.15 --read-noise 0.02` |

### 传感器 / 结构噪声

| 噪声类型 | 说明 | 关键参数 |
|---|---|---|
| `shot_read` | 散粒噪声 + 读出噪声（异方差） | `--shot-noise 0.2 --read-noise 0.02` |
| `stripe` | 条纹/带状噪声 | `--stripe-amplitude 0.12 --stripe-period 8 --stripe-direction vertical` |
| `rain` | 合成雨线（rain streaks，结构化遮挡伪影；也可写 `--noise-type derain`） | `--rain-count 40 --rain-length-min 10 --rain-length-max 24 --rain-intensity-min 0.06 --rain-intensity-max 0.16` |
| `block_bias` | 块级偏置噪声（分块固定模式） | `--block-size 8 --block-std 0.05` |
| `dead_hot` | 传感器坏点（dead/hot pixels） | `--defect-prob 0.002 --defect-hot-ratio 0.5` |
| `line_defect` | 行/列坏线（stuck rows/cols） | `--line-prob 0.01 --line-hot-ratio 0.5` |
| `rowcol_bias` | 行/列随机偏置（非周期 banding / FPN） | `--row-bias-std 0.02 --col-bias-std 0.02` |
| `quantization` | 量化噪声（ADC / bit-depth） | `--quant-bits 8 --no-quant-dither` |

### 混合噪声

| 噪声类型 | 说明 | 关键参数 |
|---|---|---|
| `mixed` | 混合噪声（shot+read → impulse → quant） | `--shot-noise 0.2 --read-noise 0.02 --impulse-prob 0.03 --quant-bits 8` |

## 快速参考

每个类别挑一个代表跑通（所有深度学习模型 variant 换成 `_tiny` 即可冒烟）：

```bash
RUN="python -m tracks.vision.lesson_10_synthetic_denoising.train"

# 传统基线
$RUN --arch bm3d:bm3d_fast --sigma 0.1
$RUN --arch median_filter:median_tiny --sigma 0.1
$RUN --arch non_local_means:nlm_fast --sigma 0.1

# 有监督 CNN（各子类各一个）
$RUN --arch dncnn:dncnn_tiny --epochs 3
$RUN --arch rdn:rdn_tiny --epochs 3
$RUN --arch resunet:resunet_tiny --epochs 3
$RUN --arch restormer:restormer_tiny --epochs 3
$RUN --arch nafnet:nafnet_tiny --epochs 3

# 噪声条件化
$RUN --arch ffdnet:ffdnet_tiny --epochs 3
$RUN --arch drunet:drunet_tiny --epochs 3

# Blind denoising
$RUN --arch cbdnet:cbdnet_tiny --epochs 3

# Noise2Noise
$RUN --arch noise2noise_unet:n2n_unet_tiny --train-mode noise2noise --epochs 3

# Blind-Spot
$RUN --arch bsn:bsn_tiny --train-mode blindspot --epochs 3

# 传感器校正
$RUN --arch dead_hot_pixel_corrector:dead_hot_tiny --noise-type dead_hot --sigma 0.1
$RUN --arch stripe_remover:stripe_remover_tiny --noise-type stripe --sigma 0.1

# 去雨
# Deraining (CNN-style)
$RUN --arch jorder:jorder_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch rescan:rescan_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch prenet:prenet_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch ddn:ddn_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch spanet:spanet_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch did_mdn:did_mdn_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch rcdnet:rcdnet_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3

# Deraining (Transformer-style)
$RUN --arch transweather:transweather_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3
$RUN --arch derainformer:derainformer_tiny --noise-type rain --rain-count 24 --rain-length-min 8 --rain-length-max 18 --rain-intensity-min 0.05 --rain-intensity-max 0.14 --epochs 3

Rain-family quick selection (`--noise-type rain`):
- CNN-style: `jorder`, `rescan`, `prenet`, `ddn`, `spanet`, `did_mdn`, `rcdnet`
- Transformer-style: `transweather`, `derainformer`

# 不同噪声模型
```

## 输出产物（统一规范）

`outputs/vision/lesson_10_synthetic_denoising/<run_name>/`

- `config.json` — 完整训练/数据配置
- `metrics.jsonl` — 逐 epoch 指标（train_mse, eval_mse, eval_psnr, lr）；传统方法输出 `metrics.json`（单次评估）
- `logs/train.log` — 训练日志
- `checkpoints/checkpoint.pt` — 模型权重

## 练习（建议）

1. **对比基线**：先跑 `bm3d:bm3d_fast`，记下 PSNR，再用 `dncnn:dncnn_9` supervised 训练 5 epoch，对比深度学习 vs 传统方法
2. **训练范式对比**：固定 `dncnn:dncnn_tiny`，分别用 `--train-mode supervised / noise2noise / blindspot` 训练，对比三种范式在相同 epoch 下的 PSNR
3. **噪声鲁棒性**：用 `gaussian_var`（随机 sigma）训练一个模型，然后用固定 sigma 测试，看模型能否泛化到不同噪声强度
4. **条件化 vs 非条件化**：对比 `dncnn`（不感知 sigma）和 `ffdnet`（拼接 sigma map）在 `gaussian_var` 噪声下的表现
5. **结构盲点 vs 训练盲点**：对比 `bsn:bsn_tiny --train-mode supervised` 和 `dncnn:dncnn_tiny --train-mode blindspot`，理解"结构保证"和"训练技巧"两条路的区别
6. **噪声模型探索**：用 `stripe` 噪声训练，观察方向性噪声对不同模型的影响
7. **传统滤波器横评**：对同一张噪声图，依次跑 `median_filter`、`bilateral_filter`、`non_local_means`、`bm3d`，对比 PSNR 和运行时间
8. **传感器缺陷修复**：用 `--noise-type dead_hot` 生成带坏点的数据，分别用 `dead_hot_pixel_corrector`（传统）和 `dncnn`（学习）去处理，对比效果
9. **U-Net 变体对比**：用 `resunet` / `attention_unet` / `unetpp` / `unet3plus` 四种 U-Net 变体在同一数据上训练，对比结构差异带来的 PSNR 变化
10. **混合噪声挑战**：用 `--noise-type mixed` 训练和测试，看哪种模型对复杂真实噪声最鲁棒

## 代码位置

- 数据：`tracks/vision/lesson_10_synthetic_denoising/data.py`
- 模型分发：`tracks/vision/lesson_10_synthetic_denoising/model.py`
- 训练：`tracks/vision/lesson_10_synthetic_denoising/train.py`
- 损失函数：`tracks/vision/lesson_10_synthetic_denoising/losses.py`
- 噪声类型注册：`tracks/vision/lesson_10_synthetic_denoising/noise_types.py`
- 算法实现：`dlhub/vision/denoising/*.py`（每个算法族一个文件）
