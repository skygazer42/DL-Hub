---
icon: material/paw
---

# Model Zoo 总览

全领域统一模型动物园 --- 纯 PyTorch 本地实现，无需下载预训练权重，**8 600+ 本地架构 ID** 一行切换（实测 8626，`python scripts/project_stats.py --json` 可复核）。

---

## 规模一览

<div class="grid cards" markdown>

-   :material-eye:{ .lg .middle } **Vision**

    ---

    **791** Architecture IDs / 220 个 backbone 模块

    [:octicons-arrow-right-24: Vision Zoo](vision-zoo.md)

-   :material-text:{ .lg .middle } **NLP**

    ---

    **814** Architecture IDs / 49 算法族

    [:octicons-arrow-right-24: NLP Zoo](nlp-zoo.md)

-   :material-cube-outline:{ .lg .middle } **Point Cloud**

    ---

    **64** Architecture IDs / 30 算法族

    [:octicons-arrow-right-24: Point Cloud Zoo](pointcloud-zoo.md)

-   :material-image-text:{ .lg .middle } **Multimodal (VLM)**

    ---

    **210** Architecture IDs / 70 算法族

    [:octicons-arrow-right-24: VLM Zoo](vlm-zoo.md)

-   :material-creation:{ .lg .middle } **Generative**

    ---

    GAN **44 族 / 132 IDs** + Diffusion **32 族 / 96 IDs**

    [:octicons-arrow-right-24: Generative Zoo](generative-zoo.md)

-   :material-server-network:{ .lg .middle } **Federated Learning**

    ---

    **76** 联邦策略 / 228 Architecture IDs

    [:octicons-arrow-right-24: Federated Zoo](federated-zoo.md)

</div>

---

## Zoo 子系统总览（22 个子系统）

| 领域 | 子系统 | 算法族 | CLI 脚本 |
|:-----|:------|:-------|:---------|
| Vision | Backbones | 220 模块 / 791 IDs | `scripts/vision_zoo.py` |
| Vision | Detection (2D) | 132 | `scripts/detection_zoo.py` |
| Vision | Instance Segmentation | 50 | `scripts/instance_segmentation_zoo.py` |
| Vision | Panoptic Segmentation | 50 | `scripts/panoptic_segmentation_zoo.py` |
| Vision | Lane Detection | 44 | `scripts/lane_detection_zoo.py` |
| Vision | Co-segmentation | 26 | `scripts/co_segmentation_zoo.py` |
| Vision | Fine-Grained Recognition | 112 | `scripts/fine_grained_recognition_zoo.py` |
| Vision | Action Recognition | 62 | `scripts/action_recognition_zoo.py` |
| Vision | MOT (2D) | 100 | `scripts/mot_zoo.py` |
| NLP | Text Encoders | 49 族 / 814 IDs | `scripts/nlp_zoo.py` |
| Point Cloud | Backbones | 30 族 / 64 IDs | `scripts/pointcloud_zoo.py` |
| Point Cloud | 3D Detection | 60 | `scripts/detection3d_zoo.py` |
| Point Cloud | 3D Segmentation | 60 | `scripts/segmentation3d_zoo.py` |
| Point Cloud | 3D Instance Seg | 50 | `scripts/instance_segmentation3d_zoo.py` |
| Point Cloud | 3D Tracking | 140 | `scripts/tracking3d_zoo.py` |
| Point Cloud | Gaussian Splatting | 10 | `dlhub/pointcloud/gaussian_splatting_zoo.py` |
| Multimodal | VLM | 70 | `scripts/vlm_zoo.py` |
| Multimodal | Prompt Learning | 10 | `dlhub/multimodal/prompt_learning_zoo.py` |
| Vision | New Directions Batch XIII | 80 | `dlhub/vision/*_zoo.py` |
| Generative | GAN | 44 | `scripts/gan_zoo.py` |
| Generative | Diffusion | 32 | `scripts/diffusion_zoo.py` |
| Federated | FL Strategies | 76 | `scripts/federated_zoo.py` |

!!! info "统计说明"

    数字为写作时实测值（`--list` 输出），实际数量随版本迭代持续增长。
    没有独立 `scripts/` CLI 的子系统（Gaussian Splatting / Prompt Learning / New Directions）
    直接给出 `dlhub/` 包内 zoo 模块路径。
    此外还有 100+ 个 Research Direction 子领域（每个 10 族），
    详见 [Research Directions](research-directions.md)。

---

## 设计原则

所有 Zoo 遵循相同的设计模式：

### 一文件一算法族

每个算法族（如 ResNet、ViT）对应一个独立 Python 文件，包含所有变体的构建逻辑。

```text
dlhub/
  vision/
    backbones/
      resnet.py        # ResNet-18 / 34 / 50 / 101 / 152 …
      vit.py           # ViT-Ti / S / B / L / H …
      convnext.py      # ConvNeXt-T / S / B / L / XL …
    detection/         # 2D 检测算法族（一文件一族）
    mot/               # 多目标跟踪算法族
  nlp/
    algorithms/
      bert.py          # BERT-Tiny / Mini / Small / Base / Large …
```

### Lazy Import

所有算法族仅在实际构建时才触发对应文件的导入，保证启动零开销。

### 统一接口

每个领域提供 `build_local_model(arch_id, ...)` / `list_local_arches()` 统一入口：

```python
from dlhub.vision.local_zoo import build_local_model

# 构建模型只需一行（本地实现的 ID 使用 `dl:` 前缀）
model = build_local_model("dl:resnet50", in_channels=3, num_classes=10)
```

### CLI 工具

每个子系统均自带 CLI，支持三个核心操作：

=== "`--list`"

    列出所有可用架构 ID。

    ```bash
    python scripts/vision_zoo.py --list
    ```

=== "`--search`"

    模糊搜索架构 ID。

    ```bash
    python scripts/vision_zoo.py --list --search resnet
    ```

=== "`--smoke`"

    对指定架构执行前向推理 Smoke Test。

    ```bash
    python scripts/vision_zoo.py --smoke dl:resnet50
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
| [Research Directions](research-directions.md) | 100+ 研究方向子领域的包路径明细 |
