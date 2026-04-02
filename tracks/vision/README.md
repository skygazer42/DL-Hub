# Vision 轨（计算机视觉）

目标：从最小的分类任务开始，掌握数据、模型、训练、评估与实验习惯，再逐步扩展到更复杂的视觉任务。

## Lessons

- `lesson_01_mnist_lenet/`：MNIST + LeNet（入门闭环）
- `lesson_02_mnist_mlp/`：MNIST + MLP（全连接基线）
- `lesson_03_mnist_alexnet/`：MNIST + AlexNet（简化实现）
- `lesson_04_synthetic_detection_fcos/`：合成目标检测（FCOS 风格，anchor-free，纯 torch）
- `lesson_05_vit_toy_classification/`：ViT（toy 合成分类，纯 torch）
- `lesson_06_swin_toy_classification/`：Swin 风格（window attention + shifted windows，toy 合成分类，纯 torch）
- `lesson_07_toy_keypoint_regression/`：toy 关键点回归（合成数据，坐标回归 + 误差指标，纯 torch）
- `lesson_08_synthetic_segmentation_unet/`：合成分割（Tiny U-Net + torchvision 分割模型 zoo，二分类 mask，loss + IoU）
- `lesson_09_cnn_backbones_toy_classification/`：CNN 主干网络（本地实现 + torchvision zoo，toy 合成分类）
- `lesson_10_synthetic_denoising/`：图像去噪（BM3D / DnCNN / Restormer / Noise2Noise / NAFNet / SwinIR / RIDNet / FFDNet / DRUNet，toy 合成数据，回归闭环）
- `lesson_11_synthetic_instance_segmentation_yolact/`：实例分割（YOLACT-style：prototypes + coefficients，toy 合成数据）
- `lesson_12_synthetic_detection_yolo/`：合成目标检测（YOLOv1-style，grid/objectness + bbox，纯 torch）
- `lesson_13_synthetic_pedestrian_detection_fcos/`：合成行人检测（FCOS-style，anchor-free，纯 torch + local zoo `dldet:pedestrian_fcos`）
- `lesson_14_video_mot_basics/`：合成视频多目标跟踪（MOT 基础闭环，纯 torch + local zoo `mot2d:*`）
- `lesson_15_neural_style_transfer_gatys/`：风格迁移（Gatys-style neural style transfer，优化式，toy-first）
- `lesson_16_style_transfer_translation_cyclegan/`：风格迁移（CycleGAN-style 图像翻译，unpaired，toy-first）
- `lesson_17_synthetic_super_resolution/`：合成配对图像超分辨率（paired super-resolution，toy-first，纯 torch + local zoo `sr:*`）
