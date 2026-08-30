---
icon: material/cube-outline
---

# Point Cloud Zoo

> **30 个注册组 / 64 个注册 ID** --- 索引点集、图、MLP、Transformer、卷积等点云方法，
> 外加 3D 下游任务入口；注册数量不等于独立论文实现数量。

!!! warning "方法标签不自动获得保真等级"

    下表“标签所指创新”描述原方法；当前源码是否保留该机制，以[保真度审计](fidelity.md)为准。

---

## CLI 快速上手

```bash
# 列出全部 64 个注册 ID（本地实现的 ID 使用 `pc:` 前缀）
python scripts/pointcloud_zoo.py --list

# 模糊搜索
python scripts/pointcloud_zoo.py --list --search pointnet

# Smoke Test（前向推理验证）
python scripts/pointcloud_zoo.py --smoke pc:pointnet2_ssg
```

---

## Backbone 注册标签分类

| 类别 | 代表算法 | 说明 |
|:-----|:---------|:----|
| Set Models | PointNet, PointNet++, DeepSets | 基于点集的对称函数建模 |
| Graph Models | DGCNN, PointGAT, PointGCN, PointWeb | 基于图神经网络的局部关系建模 |
| MLP Models | PointMLP, PointMixer, PointNeXt | 纯 MLP 点云编码 |
| Transformer | PCT, Point Transformer, PointBERT, PointMAE | Self-Attention 用于点云 |
| Conv Models | KPConv, PointCNN, PointConv, ShellNet | 点云上的卷积操作 |
| Extra | CurveNet, GDANet, PAConv, PVCNN, RandLANet, RSCNN, SpiderCNN | 其他创新结构 |

---

## Set Models

基于对称函数直接处理无序点集。

| 方法标签 | 关键变体 | 标签所指创新 |
|:------|:---------|:---------|
| PointNet | pointnet_cls, pointnet_seg | Shared MLP + Max Pooling 全局特征提取 |
| PointNet++ | pointnet2_ssg, pointnet2_msg | 分层 Set Abstraction + 多尺度分组 |
| DeepSets | deepsets_sum, deepsets_max | 置换不变集合函数理论框架 |

!!! tip "一行构建点云模型"

    ```python
    from dlhub.pointcloud.local_zoo import build_local_model

    model = build_local_model(
        "pc:pointnet2_msg",
        in_channels=3, num_classes=40, num_points=1024,
    )
    ```

---

## Graph Models

将点云构造为图结构，利用图神经网络提取局部几何关系。

| 方法标签 | 关键变体 | 标签所指创新 |
|:------|:---------|:---------|
| DGCNN | dgcnn_cls, dgcnn_seg | Dynamic Graph CNN，EdgeConv 动态图卷积 |
| PointGAT | pointgat_base | Graph Attention 用于点云 |
| PointGCN | pointgcn_base | Graph Convolution 用于点云 |
| PointWeb | pointweb_base | Adaptive Feature Adjustment 模块 |

---

## MLP Models

纯 MLP 架构在点云上的高效实现。

| 方法标签 | 关键变体 | 标签所指创新 |
|:------|:---------|:---------|
| PointMLP | pointmlp_base, pointmlp_elite | 纯残差 MLP + Geometric Affine Module |
| PointMixer | pointmixer_base | MLP-Mixer 风格的点云编码 |
| PointNeXt | pointnext_s, pointnext_b, pointnext_l | 改进 PointNet++ 训练策略 + InvResMLP |

---

## Transformer

Self-Attention 机制应用于点云理解。

| 方法标签 | 关键变体 | 标签所指创新 |
|:------|:---------|:---------|
| PCT | pct_base | 全局 Offset-Attention |
| Point Transformer | point_transformer_v1, point_transformer_v2 | Vector Attention 向量注意力 |
| PointBERT | pointbert_base | Masked Point Modeling 预训练 |
| PointMAE | pointmae_base | Masked Autoencoder 用于点云 |

---

## Conv Models

在点云上定义卷积操作，类比 2D 卷积。

| 方法标签 | 关键变体 | 标签所指创新 |
|:------|:---------|:---------|
| KPConv | kpconv_rigid, kpconv_deform | Kernel Point Convolution 核点卷积 |
| PointCNN | pointcnn_base | X-Conv 点云卷积算子 |
| PointConv | pointconv_base | 连续卷积 + 密度重加权 |
| ShellNet | shellnet_base | ShellConv 基于球壳的卷积 |

---

## Extra

其他创新性点云架构。

| 方法标签 | 关键变体 | 标签所指创新 |
|:------|:---------|:---------|
| CurveNet | curvenet_base | 曲线分组 (Curve Grouping) 捕获结构信息 |
| GDANet | gdanet_base | 几何解耦注意力 (Geometry-Disentangled Attention) |
| PAConv | paconv_base | Position Adaptive Convolution 位置自适应卷积 |
| PVCNN | pvcnn_base, pvcnn_m2 | Point-Voxel CNN 混合表示 |
| RandLANet | randlanet_base | Random Sampling + Local Feature Aggregation |
| RSCNN | rscnn_base | Relation-Shape CNN 关系形状卷积 |
| SpiderCNN | spidercnn_base | 基于泰勒展开的参数化卷积滤波器 |

---

## 3D 下游任务子系统

### 3D Detection Zoo

> 60 个注册标签 --- 索引 Point-based / Voxel-based / Pillar-based / Multi-modal 3D 检测方法。

```bash
python scripts/detection3d_zoo.py --list
python scripts/detection3d_zoo.py --search pointpillars
python scripts/detection3d_zoo.py --smoke pcdet3d:pointpillars_tiny
```

!!! example "代表算法"

    PointPillars, VoxelNet, SECOND, PV-RCNN, CenterPoint, PointRCNN, Part-A2, VoxelNeXt, TransFusion, BEVFusion

---

### 3D Segmentation Zoo

> 60 个注册标签 --- 索引 Point / Voxel / Range-view / Fusion 3D 分割方法。

```bash
python scripts/segmentation3d_zoo.py --list
python scripts/segmentation3d_zoo.py --search randlanet
python scripts/segmentation3d_zoo.py --smoke pcseg3d:randlanet_tiny
```

!!! example "代表算法"

    Cylinder3D, MinkowskiNet, SPVCNN, SalsaNext, PolarNet, RangeNet++, SqueezeSegV3, 2DPASS

---

### 3D Instance Segmentation Zoo

> 50 个注册标签 --- 索引 Proposal-based / Grouping-based / Panoptic 3D 实例分割方法。

```bash
python scripts/instance_segmentation3d_zoo.py --list
python scripts/instance_segmentation3d_zoo.py --search pointgroup
python scripts/instance_segmentation3d_zoo.py --smoke pcinst3d:pointgroup_tiny
```

!!! example "代表算法"

    PointGroup, SSTNet, HAIS, SoftGroup, Mask3D, DyCo3D, OccuSeg

---

### 3D Tracking Zoo

> 140 个注册标签 --- 索引 LiDAR / Camera-LiDAR / Radar-LiDAR 3D 跟踪方法。

```bash
python scripts/tracking3d_zoo.py --list
python scripts/tracking3d_zoo.py --search centerpoint
python scripts/tracking3d_zoo.py --smoke pctrk3d:centerpoint_track_tiny
```

!!! example "代表算法"

    SimpleTrack, CenterPoint-Track, PC3T, ShaSTA, OGR3MOT, ImmortalTracker, PolarMOT
