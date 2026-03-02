# FAQ

## 1) 运行报错：找不到模块（ModuleNotFoundError）

建议始终在仓库根目录运行：

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

如果你只是在本地“直接运行某个子目录脚本”，确保把 repo root 加到 `PYTHONPATH`，或使用我们提供的统一入口脚本（后续会提供）。

## 2) CUDA 不可用 / 训练很慢

- 先用 CPU 跑通（`--device cpu`）
- 再检查 `nvidia-smi`（如果你在本机）
- 代码层面不要默认强制 CUDA

## 3) 数据下载失败

很多数据集下载依赖网络；lesson 会提供 `--dataset fake` / `toy` 模式用于离线自检。

## 4) 为什么不用 PyTorch Lightning / HuggingFace Trainer？

这是“学习项目”，我们倾向使用 **最少抽象**，让你看得见训练循环发生了什么。

