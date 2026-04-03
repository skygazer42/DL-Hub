---
icon: material/eye
---

# Vision Zoo

> **208 算法族 / 736 Architecture IDs** --- 覆盖从经典 CNN 到最新 Vision Transformer 的全部视觉主干网络，外加 8 个下游任务子系统。

---

## CLI 快速上手

```bash
# 列出全部 736 个架构 ID
python -m zoo.vision --list

# 模糊搜索
python -m zoo.vision --search efficientnet

# Smoke Test（前向推理验证）
python -m zoo.vision --smoke resnet50
```

---

## Backbone 架构分类

| 类别 | 代表算法 | 约计数量 |
|:-----|:---------|:---------|
| 经典 CNN | AlexNet, VGG-11/13/16/19, GoogLeNet (Inception v1-v4), ResNet-18/34/50/101/152, DenseNet-121/169/201/264 | ~60 |
| 高效网络 | MobileNet v1/v2/v3/v4, EfficientNet-B0~B7, GhostNet, ShuffleNet v1/v2 | ~80 |
| 注意力 CNN | SENet, CBAM, BAM, ECA-Net | ~50 |
| 现代 CNN | ConvNeXt v1/v2, RepVGG, RepLKNet | ~40 |
| Vision Transformer | ViT-Ti/S/B/L/H, DeiT, BEiT, Swin v1/v2, CSwin | ~120 |
| 高效 Transformer | EfficientViT, TinyViT, EdgeViT, FastViT | ~60 |
| MLP 系列 | MLP-Mixer, gMLP, ResMLP, FNet | ~50 |
| Hybrid | CoAtNet, MobileFormer, Uniformer, MaxViT | ~60 |
| 特殊结构 | CapsNet, FractalNet, HRNet, NAS-derived, Mamba-Vision | ~50 |

!!! tip "一行构建任意视觉主干"

    ```python
    from zoo.vision import build

    model = build("swin_v2_base", num_classes=1000)
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

> ~120 个算法族 --- 覆盖 Anchor-based、Anchor-free、Transformer-based 检测器。

```bash
python -m zoo.det2d --list
python -m zoo.det2d --search yolo
python -m zoo.det2d --smoke fasterrcnn_r50
```

!!! example "代表算法"

    Faster R-CNN, Cascade R-CNN, RetinaNet, FCOS, ATSS, DETR, Deformable-DETR, DINO, YOLOv3~v8, RT-DETR, Co-DETR

---

### Instance Segmentation Zoo

> 40 个算法族 --- 实例级像素分割。

```bash
python -m zoo.instseg --list
python -m zoo.instseg --search mask
python -m zoo.instseg --smoke maskrcnn_r50
```

!!! example "代表算法"

    Mask R-CNN, Cascade Mask R-CNN, PointRend, SOLOv2, CondInst, Mask2Former

---

### Panoptic Segmentation Zoo

> 40 个算法族 --- 统一语义分割与实例分割。

```bash
python -m zoo.panoptic --list
python -m zoo.panoptic --search panoptic
python -m zoo.panoptic --smoke panoptic_fpn_r50
```

!!! example "代表算法"

    Panoptic FPN, Panoptic-DeepLab, MaskFormer, Mask2Former, kMaX-DeepLab

---

### Lane Detection Zoo

> 24 个算法族 --- 车道线检测。

```bash
python -m zoo.lane --list
python -m zoo.lane --search lane
python -m zoo.lane --smoke scnn
```

!!! example "代表算法"

    SCNN, LaneNet, ERFNet-Lane, PINet, PolyLaneNet, LaneATT, GANet, CLRNet, BezierLaneNet

---

### Co-segmentation Zoo

> 6 个算法族 --- 协同分割，从多张图像中发现共同目标。

```bash
python -m zoo.coseg --list
python -m zoo.coseg --smoke coseg_base
```

!!! example "代表算法"

    CoSegNet, GroupWiseNet, DeepCoseg, CycleSegNet, SPNet, CSMG

---

### Fine-Grained Recognition Zoo

> 72 个算法族 --- 细粒度图像识别（如鸟类、车型、航空器）。

```bash
python -m zoo.finegrained --list
python -m zoo.finegrained --search bilinear
python -m zoo.finegrained --smoke bcnn
```

!!! example "代表算法"

    Bilinear-CNN, NTS-Net, MAMC, DCL, PMG, TransFG, CAL, IELT, SIM-Trans

---

### Action Recognition Zoo

> 22 个算法族 --- 视频动作识别。

```bash
python -m zoo.action --list
python -m zoo.action --search slowfast
python -m zoo.action --smoke slowfast_r50
```

!!! example "代表算法"

    C3D, I3D, SlowFast, TSN, TSM, TimeSformer, VideoSwin, MViTv2, UniFormerV2

---

### MOT Zoo

> 81 个算法族 --- 多目标跟踪。

```bash
python -m zoo.mot --list
python -m zoo.mot --search byte
python -m zoo.mot --smoke bytetrack
```

!!! example "代表算法"

    SORT, DeepSORT, ByteTrack, OC-SORT, StrongSORT, FairMOT, JDE, CenterTrack, TrackFormer, MOTRv2
