# DL-Hub Zoo Conventions Implementation Plan

**Goal:** 把 DL-Hub 的各类 Zoo（NLP / Vision / PointCloud / Restoration 等）统一到同一套“**一算法族一文件**”的工程规范，并在后续扩展算法时严格遵守该规范。

**Architecture:** 每个 Zoo 目录以“算法族文件”为最小单元：一个 `.py` 文件实现一个完整网络/算法族（`torch.nn.Module`），同族的不同规模/深度/宽度等作为 **variants** 写在同一个文件里，通过 `_VARIANTS` + `build_*` 工厂函数选择；上层通过 `family:variant` 字符串构建模型。

**Tech Stack:** Python 3.10+，PyTorch（只用 `torch`/`torch.nn`/`torch.nn.functional` 等），pytest。**禁止**依赖外部模型库（如 `torchvision.models`、`timm`、`detectron2`、`mmdet` 等）来“导入现成网络”。

---

## 0) 核心定义（必须一致）

### 0.1 “算法族（family）” vs “变种（variant）”

- **算法族（family）**：论文/方法级别的结构范式（例如 `ResNet`、`UNet`、`DETR`、`DnCNN`、`Restormer`）。
- **变种（variant）**：同一算法族内的参数化配置差异（例如深度/宽度/层数/通道数/patch size 等），**不单独算一个算法族**。

例子：
- ✅ `resnet.py` 里包含 `resnet18/resnet34/resnet50/...`（variants）
- ✅ `dncnn.py` 里包含 `dncnn_9/dncnn_17/...`
- ❌ 不要把 `resnet18.py / resnet34.py / resnet50.py` 拆成多个文件（这属于“变种拆文件”，不符合规范）

**判定原则（默认规则）**
- 只改“规模/超参/轻量化宽度乘子”→ 仍是同一 family 的 variant。
- 明显改变结构范式/训练范式/任务头（例如 YOLOv5 vs YOLOX，DETR vs Faster R-CNN）→ 视为不同 family，应该拆成不同 `.py` 文件。

### 0.2 “纯 torch 手动实现”约束

必须满足：
- 网络结构由 `torch.nn.Module` 手写搭建（Conv/Norm/Attention/Pooling 等都用 torch 原语）。
- 允许抽公共 block 到同目录 `_utils.py` / `_blocks.py`（仅 torch 依赖），但 **family 级别仍是一文件**。

禁止：
- `from torchvision.models import resnet50` / `import timm` 这类“导入现成网络”。
- 以“只写论文名 + 伪代码/空壳 forward”充数；必须可跑通随机输入。

---

## 1) 文件级规范（每个 family 文件都必须符合）

### 1.1 文件命名

- 文件名：`snake_case.py`
- family 名：通常与文件名一致（例如 `dead_hot_pixel_corrector.py` → `dead_hot_pixel_corrector`）

### 1.2 必备内容（强制）

每个 family 文件至少包含：

1. 一个可实例化的 `nn.Module`（可以是主类 + 若干内部 block 类）
2. `_VARIANTS: dict[str, dict]`（或等价结构）用于列出 variants
3. `build_<family>_<role>(...) -> nn.Module` 工厂函数（role 依目录而定）
4. `if __name__ == "__main__":` 随机数据 smoke（至少 forward；建议带 backward）

**推荐模板（以 Vision Backbone 为例）**

```py
import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(...): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

_VARIANTS = {
    "resnet18": {...},
    "resnet34": {...},
}

def build_resnet_backbone(*, in_channels: int, variant: str = "resnet18", **kw) -> nn.Module:
    ...
    return ResNet(...)

if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_resnet_backbone(in_channels=3, variant="resnet18")
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    print("ok", y.shape)
```

### 1.3 builder 命名（按目录统一）

建议每个目录统一 `build_*` 入口（便于 Lesson / Zoo 自动枚举）：

- `dlhub/vision/backbones/`：`build_<family>_classifier(...)` 或 `build_<family>_backbone(...)`
- `dlhub/vision/denoising/`：`build_<family>_denoiser(...)`
- `dlhub/vision/detection/`：`build_<family>_detector(...)`
- `dlhub/vision/segmentation/`：`build_<family>_segmentor(...)`
- `dlhub/pointcloud/*`：`build_<family>_<task>(...)`（如 `*_detector3d` / `*_segmentation3d` 等）
- `dlhub/nlp/algorithms/`：`build_<family>_classifier(...)` + `registry()`（现状基本符合）

---

## 2) 上层接入规范（Lesson / Zoo）

### 2.1 统一的 arch 字符串选择

上层统一使用：
- `--arch <family>:<variant>`

实现要求：
- 每个 Lesson 的 `model.py` 必须能解析 `family:variant`（或 `cfg.arch`+`cfg.variant`）
- `list_supported_arches()` 能枚举所有 `<family>:<variant>`（用于 CLI `--list-arch`）

示例（已在 Lesson 10 去噪里落地）：
- `tracks/vision/lesson_10_synthetic_denoising/model.py`

### 2.2 计数口径（文档必须一致）

- **算法族数（families）**：等于 family 文件数（例如去噪 61 个 family 文件）
- **架构 ID 数（arch ids / variants）**：等于所有 `_VARIANTS` 的 key 总数

README/表格里必须明确写哪个口径，避免“一个算法多个变种”被重复统计。

---

## 3) 测试与验证规范（新增算法必带）

### 3.1 单文件 smoke（强制）

每个 family 文件的 `__main__` 至少做到：
- 构建 1 个 variant
- `torch.randn(...)` forward
- 打印 shape（必要时打印 params）

### 3.2 pytest smoke（强制）

每个大类目录（detection/segmentation/denoising/backbones/pointcloud/nlp）至少有 1 个 pytest：
- 遍历所有 builders
- CPU 上 forward（必要时 backward）
- 校验输出 shape / dtype / finite loss

**验证命令（最小）**
- 全局：`pytest -q`
- 只测某类：`pytest -q tests/test_dlhub_vision_detection_algorithms.py`

---

## 4) Git 工作流（强制）

- 任何“增加算法/重构目录结构”的提交都需要：
  1) 跑 `pytest -q`（或至少跑覆盖该模块的子集测试）
  2) 通过后再 `git commit`
  3) 通过后再 `git push origin main`

---

## 5) 实施任务拆分（后续执行用）

> 下面的 Task 以“2–5 分钟一步”为粒度，用于后续按规范推进各目录扩展/重构。

### Task 1：增加“规范检查”pytest（可选但强烈推荐）

**Files:**
- Create: `tests/test_zoo_conventions_smoke.py`

**Step 1: 写一个 RED 测试（先失败）**
- 扫描指定目录（如 `dlhub/vision/backbones`, `dlhub/vision/detection`, `dlhub/vision/denoising`, `dlhub/nlp/algorithms`）
- 用 `ast` 检查每个 family 文件是否：
  - 定义了 `_VARIANTS`
  - 定义了至少 1 个 `build_` 函数
  - 包含 `if __name__ == "__main__"` block（可用 AST/文本查找）

**Step 2: 运行确认失败**
- Run: `pytest -q tests/test_zoo_conventions_smoke.py`
- Expected: FAIL（旧模块未完全对齐）

**Step 3: 分目录逐步修复直到 PASS**

**Step 4: commit**
- `git commit -m "test: add zoo conventions smoke"`

### Task 2：Vision Backbones 扩展遵循本规范

**Reference plan:**
- `docs/plans/2026-03-03-vision-backbones-100-algorithms-design.md`

**Rules:**
- 新增 backbones 必须一算法族一文件；variants 同文件内；每个文件 `__main__` 随机 forward/backward。

### Task 3：目标检测 / 语义分割 / 实例分割 / 全景分割 按本规范扩展

**Reference plans:**
- Detection: `docs/plans/2026-03-04-vision-detection-40-algorithms-plan.md`
- Panoptic: `docs/plans/2026-03-05-vision-panoptic-40-algorithms-plan.md`

**Rules:**
- 每个 detector/segmentor family 是一个文件；variants 同文件内；统一 `build_*` 命名；pytest 遍历 builders。

### Task 4：NLP Zoo 对齐检查（当前大体已符合）

**Files:**
- Audit: `dlhub/nlp/algorithms/*.py`
- Audit: `dlhub/nlp/algorithms/registry.py`

**Step 1: 统一写清“family vs variant”口径**
- 只要某个 NLP 文件仍是“多个算法族塞一个文件”，就拆分到多文件。
- 只要某个 NLP 文件把 variants 拆成多个文件，就合并回一个 family 文件。

**Step 2: 增加/补齐 `__main__` smoke（如果缺失）**

---

## 6) 开放问题（需要你确认，按一个问题一个回答推进）

1) 对于“同名但大改版”的情况（例：`resnet_v1` vs `resnet_v2` / `unet` vs `unet++`），你希望算作：
   - 同一 family 的 variants（放一个文件），还是
   - 不同 family（拆多个文件）？

