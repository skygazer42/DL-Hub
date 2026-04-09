---
icon: material/creation
---

# Generative Zoo

> **36 算法族** --- 覆盖 GAN（24 族）与 Diffusion（12 族）两大生成模型体系。

---

## GAN Zoo

### CLI 快速上手

```bash
# 列出全部 GAN 架构 ID
python -m zoo.gan --list

# 模糊搜索
python -m zoo.gan --search stylegan

# Smoke Test（前向推理验证）
python -m zoo.gan --smoke dcgan_64
```

### 架构分类总览

| 类别 | 代表算法 | 说明 |
|:-----|:---------|:----|
| 无条件 GAN | DCGAN, WGAN, WGAN-GP, LSGAN, SNGAN | 无条件图像生成，训练稳定性为核心研究方向 |
| 条件 GAN | cGAN, ACGAN, InfoGAN, Pix2Pix | 以类别标签、输入图像等条件控制生成 |
| 图像翻译 | CycleGAN, StarGAN, UNIT, MUNIT | 跨域风格迁移与图像翻译 |
| 高分辨率 | ProGAN, StyleGAN, StyleGAN2, StyleGAN3 | 渐进式训练与风格控制，达到高保真生成 |
| 轻量级 | LightGAN, FastGAN | 减少训练数据与计算量的高效 GAN |

---

### 无条件 GAN

基础对抗生成网络，侧重训练稳定性与生成质量。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| DCGAN | dcgan_32, dcgan_64, dcgan_128 | 全卷积 GAN + BatchNorm 训练稳定化 |
| WGAN | wgan_32, wgan_64 | Wasserstein 距离替代 JS 散度，缓解模式崩溃 |
| WGAN-GP | wgangp_32, wgangp_64, wgangp_128 | Gradient Penalty 替代权重裁剪 |
| LSGAN | lsgan_32, lsgan_64 | Least Squares 损失，平滑决策边界 |
| SNGAN | sngan_32, sngan_64, sngan_128 | Spectral Normalization 谱归一化稳定训练 |

### 条件 GAN

引入条件信息控制生成过程。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| cGAN | cgan_mlp, cgan_conv | 条件标签 Concatenation 控制生成 |
| ACGAN | acgan_32, acgan_64 | 辅助分类器监督 + 条件生成 |
| InfoGAN | infogan_32, infogan_64 | 最大化互信息学习解耦表示 |
| Pix2Pix | pix2pix_256, pix2pix_512 | Paired Image-to-Image Translation + PatchGAN |

### 图像翻译

跨域图像转换与风格迁移。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| CycleGAN | cyclegan_256, cyclegan_512 | 无配对图像翻译 + Cycle Consistency Loss |
| StarGAN | stargan_128, stargan_256 | 单一生成器多域翻译 |
| UNIT | unit_256 | 共享隐空间假设 + 双 VAE-GAN |
| MUNIT | munit_256 | 分离内容与风格，多模态翻译 |

### 高分辨率

渐进式训练与风格控制的高保真生成。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| ProGAN | progan_256, progan_512, progan_1024 | 渐进式分辨率增长训练 |
| StyleGAN | stylegan_256, stylegan_512, stylegan_1024 | Mapping Network + Adaptive Instance Norm |
| StyleGAN2 | stylegan2_256, stylegan2_512, stylegan2_1024 | Weight Demodulation 消除伪影 |
| StyleGAN3 | stylegan3_r_256, stylegan3_t_256 | 别名消除 (Alias-Free)，平移/旋转等变性 |

### 轻量级

面向数据稀缺或低算力场景的高效 GAN。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| LightGAN | lightgan_256 | 轻量化生成器 + Skip-Layer Excitation |
| FastGAN | fastgan_256, fastgan_512 | 少样本 (~100 张) 高效训练 |

!!! tip "一行构建 GAN"

    ```python
    from zoo.gan import build

    generator, discriminator = build("stylegan2_512")
    ```

---

## Diffusion Zoo

### CLI 快速上手

```bash
# 列出全部 Diffusion 架构 ID
python -m zoo.diffusion --list

# 模糊搜索
python -m zoo.diffusion --search ddpm

# Smoke Test（前向推理验证）
python -m zoo.diffusion --smoke ddpm_cifar10
```

### 架构分类总览

| 类别 | 代表算法 | 说明 |
|:-----|:---------|:----|
| 基础扩散 | DDPM, DDIM, Score-SDE | 扩散模型理论基础 |
| 条件扩散 | Classifier-Guided, Classifier-Free | 条件引导生成 |
| 隐空间扩散 | Latent Diffusion, Stable Diffusion | 在隐空间而非像素空间执行扩散，大幅降低计算成本 |
| 快速采样 | DPM-Solver, Consistency Models | 加速采样步数从 1000 到 1~10 步 |

---

### 基础扩散

扩散概率模型的理论基石。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| DDPM | ddpm_cifar10, ddpm_celeba, ddpm_lsun | 去噪扩散概率模型，$T$ 步加噪 + 逆向去噪 |
| DDIM | ddim_cifar10, ddim_celeba | 非马尔可夫采样，确定性生成 + 加速采样 |
| Score-SDE | score_sde_ve, score_sde_vp, score_sde_sub_vp | 随机微分方程统一框架 (VE/VP/sub-VP) |

!!! note "扩散过程核心公式"

    **前向过程** (加噪):

    $$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I})$$

    **逆向过程** (去噪):

    $$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \sigma_t^2 \mathbf{I})$$

### 条件扩散

通过引导信号控制生成方向。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| Classifier-Guided | cg_diffusion_base | 外部分类器梯度引导采样 |
| Classifier-Free | cfg_diffusion_base | 无需分类器，条件/无条件联合训练 + 引导比例 $w$ |

!!! info "Classifier-Free Guidance"

    $$\hat{\epsilon}_\theta(\mathbf{x}_t, c) = (1 + w)\,\epsilon_\theta(\mathbf{x}_t, c) - w\,\epsilon_\theta(\mathbf{x}_t, \varnothing)$$

    其中 $w$ 为引导比例，$c$ 为条件，$\varnothing$ 为空条件。

### 隐空间扩散

在低维隐空间执行扩散过程，大幅提升效率。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| Latent Diffusion | ldm_kl_4, ldm_kl_8, ldm_vq_4 | Autoencoder 压缩 + 隐空间 DDPM |
| Stable Diffusion | sd_v1_4, sd_v1_5, sd_v2_1 | LDM + CLIP 文本条件，开源文生图里程碑 |

### 快速采样

将采样步数从数百/千步压缩至极少步。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| DPM-Solver | dpm_solver_1, dpm_solver_2, dpm_solver_3, dpm_solver_pp | 高阶 ODE Solver，10~20 步高质量生成 |
| Consistency Models | cm_ct, cm_cd | 一致性映射，单步生成 + 可选多步细化 |

---

## 用法示例

=== "GAN 训练循环"

    ```python
    from zoo.gan import build

    G, D = build("wgangp_64")
    # 标准 WGAN-GP 训练循环
    for real in dataloader:
        # Discriminator step
        z = torch.randn(B, G.latent_dim)
        fake = G(z)
        d_loss = D(fake).mean() - D(real).mean() + gradient_penalty(D, real, fake)
        # Generator step
        g_loss = -D(G(z)).mean()
    ```

=== "Diffusion 采样"

    ```python
    from zoo.diffusion import build

    model = build("ddim_celeba")
    # 50 步 DDIM 采样
    samples = model.sample(batch_size=16, steps=50)
    ```
