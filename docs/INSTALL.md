# 安装与环境（建议）

## 最小环境（推荐先跑通）

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

运行冒烟：

```bash
python scripts/smoke_check.py
pytest -q
```

## Vision 轨（可选）

如果你的环境已经有可用的 PyTorch/torchvision，可以跳过。

如果你希望用 pip 快速安装（可能因平台/显卡不同而需要调整）：

```bash
python -m pip install -r requirements-vision.txt
```

`requirements-vision.txt` 里包含 `timm`（可选但推荐）：用于快速启用大量视觉主干网络（模型 zoo）。

## 其他轨道

后续会提供 `requirements-gnn.txt`、`requirements-nlp.txt` 等按轨道安装方式。
