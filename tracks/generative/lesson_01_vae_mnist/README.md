# Lesson 01：VAE（MNIST，最小实现）

这节课的目标：用最小实现跑通 **Vanilla VAE** 的训练闭环，并理解三件事：

1. 编码器输出的是一个分布（`mu`, `logvar`），不是一个点。
2. `reparameterize` 让随机采样也能反向传播。
3. loss = 重建损失 + KL 散度（以及 `beta` 的意义）。

## 运行方式

从仓库根目录运行（必须用 `-m`）：

```bash
python -m tracks.generative.lesson_01_vae_mnist.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 1
```

如果你安装了 torchvision，可以跑真实 MNIST（会下载数据）：

```bash
python -m tracks.generative.lesson_01_vae_mnist.train --dataset mnist --epochs 5
```

## 输出产物（统一规范）

`outputs/generative/lesson_01_vae_mnist/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`：生成样本（`(B, 1, 28, 28)`，取值 0–1）
- `recons.pt`：重建对比（inputs + reconstructions）
- `logs/train.log`
- `checkpoints/checkpoint.pt`

如果检测到 `torchvision`，会额外写出 `samples.png` / `recons.png`。

## 练习（建议）

1. 调 `--beta`（例如 0.1 / 1.0 / 4.0），观察重建质量与潜变量的变化。
2. 把 encoder/decoder 从 MLP 换成小 Conv（参考 `Deep_project/VAE/` 的结构，但保持 lesson 的简洁）。
3. 固定 `z` 画一条 interpolation（线性插值），看生成空间是否连续。
