# PointCloud 轨（点云）

目标：用 **toy-first** 的方式理解点云的“无序点集”特性，并把 PointNet/DGCNN 的核心直觉跑通：

- 点云是集合（set），不是序列：顺序不应该影响结果
- “局部邻域”与“全局汇聚”的差异
- 简单任务先闭环：toy 分类 → 再上更大数据与更复杂网络

## Lessons

- `lesson_01_pointnet_toy_classification/`：PointNet 分类（toy 数据：Cube vs Sphere）
- `lesson_02_dgcnn_toy_classification/`：DGCNN 分类（toy 数据：Cube vs Sphere）
- `lesson_03_pointnet2_toy_classification/`：PointNet2 分类（toy 数据：Cube vs Sphere）
- `lesson_04_pointcloud_zoo_toy_classification/`：统一入口切换多种点云 backbone（toy 数据：Cube vs Sphere）
- `lesson_05_pointnet_toy_partseg/`：PointNet 点云 part segmentation（toy 数据：Cube part vs Sphere part）
- `lesson_06_dgcnn_toy_partseg/`：DGCNN 点云 part segmentation（toy 数据：Cube part vs Sphere part）
- `lesson_07_pointnet_toy_reconstruction/`：PointNet AutoEncoder 点云重建（toy 数据：noisy → clean，Chamfer distance）
- `lesson_08_pointcloud_partseg_zoo_toy/`：统一入口切换 part segmentation 模型（PointNet / DGCNN）
- `lesson_09_pointcloud_selfsupervised_simclr/`：点云自监督（SimCLR 对比学习，toy 数据）
- `lesson_10_pointcloud_selfsupervised_pointmae/`：点云自监督（PointMAE-style masked modeling，toy 数据）
- `lesson_11_pointcloud_selfsupervised_byol/`：点云自监督（BYOL，无负样本 bootstrap，toy 数据）
- `lesson_12_pointcloud_selfsupervised_vicreg/`：点云自监督（VICReg，去相关正则，toy 数据）
- `lesson_13_pointcloud_ssl_linear_probe/`：自监督表征评估（Linear probe / Fine-tune，toy 数据）
- `lesson_14_pointcloud_selfsupervised_moco/`：点云自监督（MoCo v2 + queue 负样本，toy 数据）
- `lesson_15_pointcloud_selfsupervised_simsiam/`：点云自监督（SimSiam，无负样本/无动量编码器，toy 数据）
- `lesson_16_pointcloud_selfsupervised_swav/`：点云自监督（SwAV + Sinkhorn balanced assignment，toy 数据）
- `lesson_17_pointcloud_selfsupervised_barlowtwins/`：点云自监督（Barlow Twins 冗余约束/去相关，toy 数据）
- `lesson_18_pointcloud_selfsupervised_dino/`：点云自监督（DINO teacher/student + centering，toy 数据）
- `lesson_19_pointcloud_selfsupervised_dinov2/`：点云自监督（DINOv2-style：DINO + iBOT patch，toy 数据）
- `lesson_20_pointcloud_selfsupervised_ijepa/`：点云自监督（I-JEPA-style masked prediction，toy 数据）
- `lesson_21_pointcloud_selfsupervised_msn/`：点云自监督（MSN-style masked distillation + prototype balance，toy 数据）
- `lesson_22_pointcloud_selfsupervised_data2vec/`：点云自监督（data2vec-style masked representation regression，toy 数据）
- `lesson_23_pointcloud_selfsupervised_ressl/`：点云自监督（ReSSL-style relational distillation + queue，toy 数据）
- `lesson_24_toy_pointcloud_completion/`：点云补全（partial → complete，Chamfer distance，toy 数据）
- `lesson_25_toy_scene_flow_estimation/`：点云场景流（双帧 per-point motion regression，toy 数据）
- `lesson_26_toy_gaussian_splatting/`：toy Gaussian Splatting（点到高斯参数映射 + 可微 splat 渲染，toy 数据）
- `lesson_27_toy_3d_object_detection/`：toy 3D 目标检测（点云到 3D box + class 预测，toy 数据）
- `lesson_28_toy_3d_semantic_segmentation/`：toy 3D 语义分割（per-point 类别预测，toy 数据）
- `lesson_29_toy_3d_instance_segmentation/`：toy 3D 实例分割（per-point instance ID 预测，toy 数据）
- `lesson_30_toy_3d_object_tracking/`：toy 3D 目标跟踪（跨帧轨迹状态回归，toy 数据）
- `lesson_31_toy_open_vocabulary_3d/`：toy Open-Vocabulary 3D（文本条件 3D 识别/grounding，toy 数据）
- `lesson_32_toy_pointcloud_forecasting/`：点云预测（历史序列到未来点云轨迹，toy 数据）
- `lesson_33_toy_pointcloud_anomaly_detection/`：点云异常检测（重建残差 + 异常得分，toy 数据）
- `lesson_34_toy_pointcloud_upsampling/`：点云上采样（稀疏点集到稠密点集恢复，Chamfer distance，toy 数据）
- `lesson_35_toy_shape_correspondence_3d/`：三维形状对应（source/target 点级匹配学习，correspondence 监督，toy 数据）
- `lesson_36_toy_pointcloud_registration/`：点云配准（source/target 刚体对齐，pose6d 回归，toy 数据）
