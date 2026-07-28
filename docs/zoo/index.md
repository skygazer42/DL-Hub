---
icon: material/paw
---

# Model Zoo 总览

全领域统一模型注册表 --- 纯 PyTorch 本地实现，无需下载预训练权重，**8 600+ 可构建注册 ID** 一行切换（当前实测 8611，`python scripts/project_stats.py --json` 可复核）。注册数量描述接口覆盖，不等同于独立论文复现数量；命名、数据和验证边界见[实现契约](../implementation-contract.md)，机制对齐情况见 [Model Zoo 保真度审计](fidelity.md)。

---

## 规模一览

<div class="grid cards" markdown>

-   :material-eye:{ .lg .middle } **Vision**

    ---

    **791** 注册 ID / 220 个 backbone 模块

    [:octicons-arrow-right-24: Vision Zoo](vision-zoo.md)

-   :material-text:{ .lg .middle } **NLP**

    ---

    **814** 注册 ID / 49 个注册族

    [:octicons-arrow-right-24: NLP Zoo](nlp-zoo.md)

-   :material-cube-outline:{ .lg .middle } **Point Cloud**

    ---

    **64** 注册 ID / 30 个注册族

    [:octicons-arrow-right-24: Point Cloud Zoo](pointcloud-zoo.md)

-   :material-image-text:{ .lg .middle } **Multimodal (VLM)**

    ---

    **210** 注册 ID / 70 个注册族

    [:octicons-arrow-right-24: VLM Zoo](vlm-zoo.md)

-   :material-creation:{ .lg .middle } **Generative**

    ---

    GAN **44 族 / 132 IDs** + Diffusion **32 族 / 96 IDs**

    [:octicons-arrow-right-24: Generative Zoo](generative-zoo.md)

-   :material-server-network:{ .lg .middle } **Federated Learning**

    ---

    **76** 联邦策略 / 228 注册 ID

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
    “注册 ID”只表示统一构建接口可识别的配置；多个 ID 可能共享通用基线，
    未经审计的实现统一标记为 `unreviewed`，不能据此推断论文机制完整。
    没有独立 `scripts/` CLI 的子系统（Gaussian Splatting / Prompt Learning / New Directions）
    直接给出 `dlhub/` 包内 zoo 模块路径。
    此外还有 100+ 个 Research Direction 子领域（每个 10 族），
    详见 [Research Directions](research-directions.md)。

---

## 设计原则

所有 Zoo 遵循相同的设计模式：

### 一文件一注册族

每个注册族通常对应一个独立 Python 文件，包含其变体构建逻辑；部分教学型注册族会共享轻量基线。文件名和注册名用于定位实现，不构成论文级复现承诺，差异见[保真度审计](fidelity.md)。

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

    列出所有可用注册 ID。

    ```bash
    python scripts/vision_zoo.py --list
    ```

=== "`--search`"

    模糊搜索注册 ID。

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
| [保真度审计](fidelity.md) | 已审计实现的机制对齐等级、证据与下一步 |
| [Vision Zoo](vision-zoo.md) | CNN、Transformer、MLP、Hybrid 等视觉主干及 8 个下游子系统 |
| [NLP Zoo](nlp-zoo.md) | Transformer、RNN、CNN、MLP 等文本编码器 |
| [Point Cloud Zoo](pointcloud-zoo.md) | 点云主干及 3D Detection / Segmentation / Tracking |
| [VLM Zoo](vlm-zoo.md) | 视觉-语言多模态模型 |
| [Generative Zoo](generative-zoo.md) | GAN 与 Diffusion 生成模型 |
| [Federated Zoo](federated-zoo.md) | 联邦学习策略 |
| [Research Directions](research-directions.md) | 100+ 研究方向子领域的包路径明细 |
