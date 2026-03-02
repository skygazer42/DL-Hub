# Lesson 02：GAN（MNIST，最小实现）

这节课的目标：用最小实现跑通 **Vanilla GAN** 的训练闭环，理解对抗训练的基本逻辑：

- 判别器 `D` 学会区分真/假
- 生成器 `G` 学会“骗过”判别器
- 两个网络交替更新，loss 不一定单调下降，但训练应该稳定、无 NaN

## 运行方式

离线冒烟（不下载数据）：

```bash
python -m tracks.generative.lesson_02_gan_mnist.train --dataset fake --epochs 1 --max-train-batches 2 --device cpu --run-name smoke
```

如果你安装了 torchvision，可以跑真实 MNIST（会下载数据）：

```bash
python -m tracks.generative.lesson_02_gan_mnist.train --dataset mnist --epochs 10
```

## 输出产物（统一规范）

`outputs/generative/lesson_02_gan_mnist/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`（以及可选的 `samples.png`）
- `logs/train.log`
- `checkpoints/checkpoint.pt`（包含 G/D 的参数）

## 练习（建议）

1. 给 `D` 加 label smoothing（例如 real label=0.9），观察稳定性。
2. 把 BCE 换成 hinge loss 或 WGAN-GP（进阶）。
3. 把结构从 MLP 换成小卷积（DCGAN 风格）。
