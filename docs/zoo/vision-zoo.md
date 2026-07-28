---
icon: material/eye
---

# Vision Zoo

> **220 个 backbone 模块 / 791 Architecture IDs** --- 覆盖从经典 CNN 到最新 Vision Transformer 的全部视觉主干网络，外加 8 个下游任务子系统。

---

## CLI 快速上手

```bash
# 列出所有可用架构（本地实现的 ID 使用 `dl:` 前缀）
python scripts/vision_zoo.py --list

# 模糊搜索
python scripts/vision_zoo.py --list --search convnext

# Smoke Test（前向推理验证）
python scripts/vision_zoo.py --smoke dl:resnet50
```

---

## Backbone 架构分类

| 类别 | 代表算法 | 约计数量 |
|:-----|:---------|:---------|
| 经典 CNN | AlexNet, VGG-11/13/16/19, GoogLeNet (Inception v1-v4), ResNet-18/34/50/101/152, DenseNet-121/169/201/264 | ~60 |
| 高效网络 | MobileNet v1/v2/v3/v4, EfficientNet v1/v2, GhostNet, ShuffleNet v1/v2 | ~80 |
| 注意力 CNN | SENet, CBAM, BAM, ECA-Net, SK-Net, CoordAtt | ~50 |
| 现代 CNN | ConvNeXt v1/v2, RepVGG, RepLKNet, HorNet, FocalNet | ~40 |
| Vision Transformer | ViT, DeiT, BEiT, Swin v2, CSwin, CaiT, CrossViT | ~120 |
| 高效 Transformer | EfficientViT, TinyViT, EdgeViT, FastViT, SwiftFormer | ~60 |
| MLP 系列 | MLP-Mixer, gMLP, ResMLP, FNet, CycleMLP, WaveMLP | ~50 |
| Hybrid | CoAtNet, MobileFormer, Uniformer, MaxViT, MobileViT | ~60 |
| 特殊结构 | CapsNet, FractalNet, HRNet, NAS 系列, Mamba | ~50 |

!!! tip "一行构建任意视觉主干"

    ```python
    from dlhub.vision.local_zoo import build_local_model

    model = build_local_model("dl:swin_v2_tiny", in_channels=3, num_classes=1000)
    ```

---

## 经典 CNN

经典卷积神经网络奠定了深度学习视觉领域的基础。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| AlexNet | alexnet | 首个大规模 CNN，ReLU + Dropout |
| VGG | vgg11, vgg13, vgg16, vgg19 (+BN) | 统一 3x3 卷积堆叠 |
| GoogLeNet | inception_v1/v2/v3/v4 | Inception Module 多尺度并行 |
| ResNet | resnet18/34/50/101/152 | Residual Connection 残差连接 |
| DenseNet | densenet121/169/201/264 | Dense Connection 密集连接 |

## 高效网络

面向移动端与边缘设备设计的轻量级架构。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| MobileNet | v1, v2, v3_small/large, v4 | Depthwise Separable Conv |
| EfficientNet | b0~b7, v2_s/m/l | Compound Scaling |
| GhostNet | ghostnet_050/100/130 | Ghost Module 廉价特征生成 |
| ShuffleNet | v1_g1/g2/g3/g4/g8, v2_x05/x10/x15/x20 | Channel Shuffle |

## Vision Transformer

基于 Self-Attention 的视觉模型已成为主流。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| ViT | vit_tiny/small/base/large/huge | Patch Embedding + Transformer |
| DeiT | deit_tiny/small/base | 数据高效训练 + Distillation Token |
| BEiT | beit_base/large | Masked Image Modeling 预训练 |
| Swin | swin_v2_tiny/small/base/large | Shifted Window Attention |
| CSwin | cswin_tiny/small/base/large | Cross-Shaped Window Attention |

## 高效 Transformer

在保持 Transformer 精度的同时降低计算成本。

| 算法族 | 关键变体 | 核心创新 |
|:------|:---------|:---------|
| EfficientViT | efficientvit_b0~b3, m0~m5 | Cascaded Group Attention |
| TinyViT | tinyvit_5m/11m/21m | 知识蒸馏 + 小模型设计 |
| EdgeViT | edgevit_xxs/xs/s | Local-Global-Local 交替 |
| FastViT | fastvit_t8/t12/s12/sa12/sa24/sa36/ma36 | RepMixer + Structural Reparameterization |

---

## 下游任务子系统

### Detection Zoo 2D

> 132 个算法族 --- 覆盖 Anchor-based、Anchor-free、Transformer-based、轻量级检测器。

```bash
python scripts/detection_zoo.py --list
python scripts/detection_zoo.py --search fcos
python scripts/detection_zoo.py --smoke dldet:fcos_tiny
```

!!! example "代表算法"

    Faster R-CNN, Cascade R-CNN, RetinaNet, FCOS, ATSS, DETR, Deformable-DETR, DINO, YOLOv3~v8, RT-DETR, Co-DETR

---

### Instance Segmentation Zoo

> 50 个算法族 --- 实例级像素分割。

```bash
python scripts/instance_segmentation_zoo.py --list
python scripts/instance_segmentation_zoo.py --search mask
python scripts/instance_segmentation_zoo.py --smoke dlinst:mask_rcnn_tiny
```

!!! example "代表算法"

    Mask R-CNN, Cascade Mask R-CNN, PointRend, SOLOv2, CondInst, Mask2Former

---

### Panoptic Segmentation Zoo

> 50 个算法族 --- 统一语义分割与实例分割。

```bash
python scripts/panoptic_segmentation_zoo.py --list
python scripts/panoptic_segmentation_zoo.py --search panoptic
python scripts/panoptic_segmentation_zoo.py --smoke dlpan:panoptic_fpn_tiny
```

!!! example "代表算法"

    Panoptic FPN, Panoptic-DeepLab, MaskFormer, Mask2Former, kMaX-DeepLab

---

### Lane Detection Zoo

> 44 个算法族 --- 车道线检测，Anchor / Parametric / Segmentation / Keypoint / Transformer 五大范式。

```bash
python scripts/lane_detection_zoo.py --list
python scripts/lane_detection_zoo.py --search laneatt
python scripts/lane_detection_zoo.py --smoke dllane:laneatt_tiny
```

!!! example "代表算法"

    SCNN, LaneNet, ERFNet-Lane, PINet, PolyLaneNet, LaneATT, GANet, CLRNet, BezierLaneNet

---

### Co-segmentation Zoo

> 26 个算法族 --- 协同分割（Group / Pair 级别），从多张图像中发现共同目标。

```bash
python scripts/co_segmentation_zoo.py --list
python scripts/co_segmentation_zoo.py --smoke coseg:clip_coseg_tiny
```

!!! example "代表算法"

    CoSegNet, GroupWiseNet, DeepCoseg, CycleSegNet, SPNet, CSMG

---

### Fine-Grained Recognition Zoo

> 112 个算法族 --- 细粒度图像识别（FGVC），Bilinear / Part-based / Transformer / Prompt / CLIP / MLLM reasoning；本地实现无需下载权重，具体实现等级以保真度审计为准。

```bash
python scripts/fine_grained_recognition_zoo.py --list
python scripts/fine_grained_recognition_zoo.py --search transfg
python scripts/fine_grained_recognition_zoo.py --smoke dlfgvc:fine_r1_tiny
```

> 时间线与方法说明见 `dlhub/vision/fine_grained_recognition/README.md`

!!! example "代表算法"

    Bilinear-CNN, NTS-Net, MAMC, DCL, PMG, TransFG, CAL, IELT, SIM-Trans

---

### Action Recognition Zoo

> 62 个算法族 --- 行为识别，Video (NCTHW，`dlactv:` 前缀) + Skeleton (NCTV，`dlacts:` 前缀)；本地实现无需下载权重，具体实现等级以保真度审计为准。

```bash
python scripts/action_recognition_zoo.py --list
python scripts/action_recognition_zoo.py --search stgcn
python scripts/action_recognition_zoo.py --smoke dlactv:c3d_tiny
python scripts/action_recognition_zoo.py --smoke dlacts:stgcn_tiny
```

> 时间线与方法说明见 `dlhub/vision/action_recognition/README.md`

!!! example "代表算法"

    C3D, I3D, SlowFast, TSN, TSM, TimeSformer, VideoSwin, MViTv2, UniFormerV2

---

### MOT Zoo

> 100 个算法族 --- 2D 单相机多目标跟踪，每族 `tiny/small/base` 三档变体。

```bash
python scripts/mot_zoo.py --list
python scripts/mot_zoo.py --search bytetrack
python scripts/mot_zoo.py --timeline
python scripts/mot_zoo.py --smoke mot2d:sort_tiny
```

除通用的 `--list / --search / --smoke` 外，MOT Zoo 还内置选型与批量训练工具链：

```bash
# 按场景推荐算法（realtime / occlusion 等 profile）
python scripts/mot_zoo.py --recommend realtime --top-k 8 --variant tiny
python scripts/mot_zoo.py --recommend occlusion --top-k 8 --variant tiny --emit-train-cmds

# 直接批量执行推荐算法的训练命令
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --skip-existing
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --summary-only
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --rank-by loss
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-leaderboard outputs/vision/mot_leaderboard.json
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir outputs/vision/mot_artifacts
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir auto
```

> 组别、选型建议与完整族列表见 `dlhub/vision/mot/README.md`

!!! example "代表算法"

    SORT, DeepSORT, ByteTrack, OC-SORT, StrongSORT, FairMOT, JDE, CenterTrack, TrackFormer, MOTRv2
