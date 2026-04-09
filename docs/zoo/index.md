---
icon: material/paw
---

# Model Zoo 总览

全领域统一模型动物园 --- 纯 PyTorch 本地实现，无需下载预训练权重，**2 500+ 架构 ID** 一行切换。

---

## 规模一览

<div class="grid cards" markdown>

-   :material-eye:{ .lg .middle } **Vision**

    ---

    **736** Architecture IDs / 208 算法族

    [:octicons-arrow-right-24: Vision Zoo](vision-zoo.md)

-   :material-text:{ .lg .middle } **NLP**

    ---

    **813** Architecture IDs / 49 算法族

    [:octicons-arrow-right-24: NLP Zoo](nlp-zoo.md)

-   :material-cube-outline:{ .lg .middle } **Point Cloud**

    ---

    **64** Architecture IDs / 30 算法族

    [:octicons-arrow-right-24: Point Cloud Zoo](pointcloud-zoo.md)

-   :material-image-text:{ .lg .middle } **Multimodal (VLM)**

    ---

    **20** 算法族

    [:octicons-arrow-right-24: VLM Zoo](vlm-zoo.md)

-   :material-creation:{ .lg .middle } **Generative**

    ---

    **36** 算法族 (GAN 24 + Diffusion 12)

    [:octicons-arrow-right-24: Generative Zoo](generative-zoo.md)

-   :material-server-network:{ .lg .middle } **Federated Learning**

    ---

    **36** 联邦策略

    [:octicons-arrow-right-24: Federated Zoo](federated-zoo.md)

</div>

---

## 21 个 Zoo 子系统

| 领域 | 子系统 | 算法族数量 | CLI 脚本 |
|:-----|:------|:----------|:---------|
| Vision | Backbones | 208 族 / 736 IDs | `python -m zoo.vision` |
| Vision | Detection 2D | ~120 | `python -m zoo.det2d` |
| Vision | Instance Segmentation | 40 | `python -m zoo.instseg` |
| Vision | Panoptic Segmentation | 40 | `python -m zoo.panoptic` |
| Vision | Lane Detection | 24 | `python -m zoo.lane` |
| Vision | Co-segmentation | 6 | `python -m zoo.coseg` |
| Vision | Fine-Grained Recognition | 72 | `python -m zoo.finegrained` |
| Vision | Action Recognition | 22 | `python -m zoo.action` |
| Vision | MOT | 81 | `python -m zoo.mot` |
| NLP | Text Encoders | 49 族 / 813 IDs | `python -m zoo.nlp` |
| Point Cloud | Backbones | 30 族 / 64 IDs | `python -m zoo.pc` |
| Point Cloud | 3D Detection | 40 | `python -m zoo.det3d` |
| Point Cloud | 3D Segmentation | 40 | `python -m zoo.seg3d` |
| Point Cloud | 3D Instance Segmentation | 30 | `python -m zoo.instseg3d` |
| Point Cloud | 3D Tracking | 131 | `python -m zoo.track3d` |
| Multimodal | VLM | 20 | `python -m zoo.vlm` |
| Generative | GAN | 24 | `python -m zoo.gan` |
| Generative | Diffusion | 12 | `python -m zoo.diffusion` |
| Federated | FL Strategies | 36 | `python -m zoo.fl` |

!!! info "统计说明"

    上表中 "~" 前缀表示近似值，实际数量随版本迭代持续增长。

---

## 设计原则

### 一文件一算法族

每个算法族（如 ResNet、ViT）对应一个独立 Python 文件，包含所有变体的构建逻辑。

```text
zoo/
  vision/
    resnet.py          # ResNet-18 / 34 / 50 / 101 / 152 …
    vit.py             # ViT-Ti / S / B / L / H …
    convnext.py        # ConvNeXt-T / S / B / L / XL …
  nlp/
    bert.py            # BERT-Tiny / Mini / Small / Base / Large …
```

### Lazy Import

所有算法族在 `import zoo` 时 **不会** 立即加载。仅在调用 `build()` 时才触发对应文件的导入，保证启动零开销。

### 统一接口

```python
from zoo.vision import build

# 构建模型只需一行
model = build("resnet50", num_classes=1000)
```

所有子系统共享相同签名：

```python
def build(arch_id: str, *, num_classes: int = 1000, **kwargs) -> nn.Module:
    ...
```

### CLI 工具

每个子系统均自带 CLI，支持三个核心操作：

=== "`--list`"

    列出所有可用架构 ID。

    ```bash
    python -m zoo.vision --list
    ```

=== "`--search`"

    模糊搜索架构 ID。

    ```bash
    python -m zoo.vision --search resnet
    ```

=== "`--smoke`"

    对指定架构执行前向推理 Smoke Test。

    ```bash
    python -m zoo.vision --smoke resnet50
    ```

---

## 快速导航

| 页面 | 说明 |
|:-----|:----|
| [Vision Zoo](vision-zoo.md) | CNN、Transformer、MLP、Hybrid 等视觉主干及 8 个下游子系统 |
| [NLP Zoo](nlp-zoo.md) | Transformer、RNN、CNN、MLP 等文本编码器 |
| [Point Cloud Zoo](pointcloud-zoo.md) | 点云主干及 3D Detection / Segmentation / Tracking |
| [VLM Zoo](vlm-zoo.md) | 视觉-语言多模态模型 |
| [Generative Zoo](generative-zoo.md) | GAN 与 Diffusion 生成模型 |
| [Federated Zoo](federated-zoo.md) | 联邦学习策略 |
