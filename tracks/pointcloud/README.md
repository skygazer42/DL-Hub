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
