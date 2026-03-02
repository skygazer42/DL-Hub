# Generative 轨（生成模型）

目标：用 **toy-first** 的方式建立生成建模直觉，并且保持与仓库其它轨道一致的训练/输出规范：

- 能跑通（CPU 也能快速冒烟）
- 能看懂（代码短、注释少但结构清晰）
- 能改动（每节课有练习建议）
- 能验收（输出目录统一，`scripts/smoke_check.py` 覆盖）

> 说明：本轨道默认不强依赖 `torchvision`。如果你选择 `--dataset mnist` 才需要安装 `torchvision`。

## Lessons

- `lesson_01_vae_mnist/`：Vanilla VAE（最小实现，支持 `--dataset fake`）
- `lesson_02_gan_mnist/`：Vanilla GAN（最小实现，MLP 结构，支持 `--dataset fake`）
