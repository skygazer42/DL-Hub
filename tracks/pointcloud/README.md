# PointCloud 轨（点云）

目标：用 **compact-first** 的方式理解点云的“无序点集”特性，并把 PointNet/DGCNN 的核心直觉跑通：

- 点云是集合（set），不是序列：顺序不应该影响结果
- “局部邻域”与“全局汇聚”的差异
- 简单任务先闭环：compact 分类 → 再上更大数据与更复杂网络

## Lessons

- `lesson_01_pointnet_compact_classification/`：PointNet 分类（compact 数据：Cube vs Sphere）
- `lesson_02_dgcnn_compact_classification/`：DGCNN 分类（compact 数据：Cube vs Sphere）
- `lesson_03_pointnet2_compact_classification/`：PointNet2 分类（compact 数据：Cube vs Sphere）
- `lesson_04_pointcloud_zoo_compact_classification/`：统一入口切换多种点云 backbone（compact 数据：Cube vs Sphere）
- `lesson_05_pointnet_compact_partseg/`：PointNet 点云 part segmentation（compact 数据：Cube part vs Sphere part）
- `lesson_06_dgcnn_compact_partseg/`：DGCNN 点云 part segmentation（compact 数据：Cube part vs Sphere part）
- `lesson_07_pointnet_compact_reconstruction/`：PointNet AutoEncoder 点云重建（compact 数据：noisy → clean，Chamfer distance）
- `lesson_08_pointcloud_partseg_zoo_compact/`：统一入口切换 part segmentation 模型（PointNet / DGCNN）
- `lesson_09_pointcloud_selfsupervised_simclr/`：点云自监督（SimCLR 对比学习，compact 数据）
- `lesson_10_pointcloud_selfsupervised_pointmae/`：点云自监督（PointMAE-style masked modeling，compact 数据）
- `lesson_11_pointcloud_selfsupervised_byol/`：点云自监督（BYOL，无负样本 bootstrap，compact 数据）
- `lesson_12_pointcloud_selfsupervised_vicreg/`：点云自监督（VICReg，去相关正则，compact 数据）
- `lesson_13_pointcloud_ssl_linear_probe/`：自监督表征评估（Linear probe / Fine-tune，compact 数据）
- `lesson_14_pointcloud_selfsupervised_moco/`：点云自监督（MoCo v2 + queue 负样本，compact 数据）
- `lesson_15_pointcloud_selfsupervised_simsiam/`：点云自监督（SimSiam，无负样本/无动量编码器，compact 数据）
- `lesson_16_pointcloud_selfsupervised_swav/`：点云自监督（SwAV + Sinkhorn balanced assignment，compact 数据）
- `lesson_17_pointcloud_selfsupervised_barlowtwins/`：点云自监督（Barlow Twins 冗余约束/去相关，compact 数据）
- `lesson_18_pointcloud_selfsupervised_dino/`：点云自监督（DINO teacher/student + centering，compact 数据）
- `lesson_19_pointcloud_selfsupervised_dinov2/`：点云自监督（DINOv2-style：DINO + iBOT patch，compact 数据）
- `lesson_20_pointcloud_selfsupervised_ijepa/`：点云自监督（I-JEPA-style masked prediction，compact 数据）
- `lesson_21_pointcloud_selfsupervised_msn/`：点云自监督（MSN-style masked distillation + prototype balance，compact 数据）
- `lesson_22_pointcloud_selfsupervised_data2vec/`：点云自监督（data2vec-style masked representation regression，compact 数据）
- `lesson_23_pointcloud_selfsupervised_ressl/`：点云自监督（ReSSL-style relational distillation + queue，compact 数据）
- `lesson_24_compact_pointcloud_completion/`：点云补全（partial → complete，Chamfer distance，compact 数据）
- `lesson_25_compact_scene_flow_estimation/`：点云场景流（双帧 per-point motion regression，compact 数据）
- `lesson_26_compact_gaussian_splatting/`：compact Gaussian Splatting（点到高斯参数映射 + 可微 splat 渲染，compact 数据）
- `lesson_27_compact_3d_object_detection/`：compact 3D 目标检测（点云到 3D box + class 预测，compact 数据）
- `lesson_28_compact_3d_semantic_segmentation/`：compact 3D 语义分割（per-point 类别预测，compact 数据）
- `lesson_29_compact_3d_instance_segmentation/`：compact 3D 实例分割（per-point instance ID 预测，compact 数据）
- `lesson_30_compact_3d_object_tracking/`：compact 3D 目标跟踪（跨帧轨迹状态回归，compact 数据）
- `lesson_31_compact_open_vocabulary_3d/`：compact Open-Vocabulary 3D（文本条件 3D 识别/grounding，compact 数据）
- `lesson_32_compact_pointcloud_forecasting/`：点云预测（历史序列到未来点云轨迹，compact 数据）
- `lesson_33_compact_pointcloud_anomaly_detection/`：点云异常检测（重建残差 + 异常得分，compact 数据）
- `lesson_34_compact_pointcloud_upsampling/`：点云上采样（稀疏点集到稠密点集恢复，Chamfer distance，compact 数据）
- `lesson_35_compact_shape_correspondence_3d/`：三维形状对应（source/target 点级匹配学习，correspondence 监督，compact 数据）
- `lesson_36_compact_pointcloud_registration/`：点云配准（source/target 刚体对齐，pose6d 回归，compact 数据）
