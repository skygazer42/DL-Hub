---
icon: material/compass-outline
---

# Research Directions / 研究方向

除各领域主 Zoo 外，`dlhub/` 还收录了 100+ 个研究方向子领域，每个方向为一个独立包
（当前每个方向 10 个算法族，本地实现无需下载权重）。下表按批次列出全部方向与包路径；
注册名只表示统一入口，具体实现等级以保真度审计为准。

!!! tip "如何使用"

    大部分方向在包的上级目录附带 `<方向名>_zoo.py` 构建入口
    （如 `dlhub/vision/blur_detection_zoo.py`、`dlhub/generative/world_models_zoo.py`），
    遵循与主 Zoo 相同的一文件一算法族设计模式；
    其余方向以独立算法族包的形式提供，由对应 track lesson 与测试直接引用。

---

## Research Directions / 研究方向（一）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| ReID / 行人重识别 | 10 | `dlhub/vision/reid/` |
| OCR / 文字识别 | 10 | `dlhub/vision/ocr/` |
| Depth Estimation / 深度估计 | 10 | `dlhub/vision/depth_estimation/` |
| Dehazing / 去雾 | 10 | `dlhub/vision/dehazing/` |
| Deblurring / 去模糊 | 10 | `dlhub/vision/deblurring/` |
| Saliency Detection / 显著性检测 | 10 | `dlhub/vision/saliency_detection/` |
| Anomaly Detection / 异常检测 | 10 | `dlhub/vision/anomaly_detection/` |
| Image Retrieval / 图像检索 | 10 | `dlhub/vision/image_retrieval/` |
| Medical Segmentation / 医学分割 | 10 | `dlhub/vision/medical_segmentation/` |
| Remote Sensing Detection / 遥感检测 | 10 | `dlhub/vision/remote_sensing_detection/` |

## Research Directions / 研究方向（二）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| HOI Detection / 人物交互检测 | 10 | `dlhub/vision/hoi_detection/` |
| Weakly Supervised Detection / 弱监督检测 | 10 | `dlhub/vision/weakly_supervised_detection/` |
| Weakly Supervised Segmentation / 弱监督分割 | 10 | `dlhub/vision/weakly_supervised_segmentation/` |
| Video Object Segmentation / 视频目标分割 | 10 | `dlhub/vision/video_object_segmentation/` |
| Crowd Counting / 人群计数 | 10 | `dlhub/vision/crowd_counting/` |
| Face Detection / 人脸检测 | 10 | `dlhub/vision/face_detection/` |
| Face Alignment / 人脸对齐 | 10 | `dlhub/vision/face_alignment/` |
| Human Pose Estimation / 人体姿态估计 | 10 | `dlhub/vision/human_pose_estimation/` |
| Video Restoration / 视频修复 | 10 | `dlhub/vision/video_restoration/` |
| Geo-localization / 地理定位 | 10 | `dlhub/vision/geo_localization/` |

## Research Directions / 研究方向（三）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Text Detection / 文本检测 | 10 | `dlhub/vision/text_detection/` |
| Text Recognition / 文本识别 | 10 | `dlhub/vision/text_recognition/` |
| Video Instance Segmentation / 视频实例分割 | 10 | `dlhub/vision/video_instance_segmentation/` |
| 3D Pose Estimation / 3D 姿态估计 | 10 | `dlhub/vision/pose_estimation_3d/` |
| 6D Pose Estimation / 6D 姿态估计 | 10 | `dlhub/vision/sixd_pose_estimation/` |
| Face Anti-Spoofing / 活体检测 | 10 | `dlhub/vision/face_anti_spoofing/` |
| Facial Expression Recognition / 表情识别 | 10 | `dlhub/vision/facial_expression_recognition/` |
| Person Attribute Recognition / 行人属性识别 | 10 | `dlhub/vision/person_attribute_recognition/` |
| License Plate Recognition / 车牌识别 | 10 | `dlhub/vision/license_plate_recognition/` |
| Sketch Retrieval / 草图检索 | 10 | `dlhub/vision/sketch_retrieval/` |

## Research Directions / 研究方向（四）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Matting / 图像抠图 | 10 | `dlhub/vision/image_matting/` |
| Image Harmonization / 图像协调 | 10 | `dlhub/vision/image_harmonization/` |
| Image Inpainting / 图像修复 | 10 | `dlhub/vision/image_inpainting/` |
| Image Fusion / 图像融合 | 10 | `dlhub/vision/image_fusion/` |
| Image Stitching / 图像拼接 | 10 | `dlhub/vision/image_stitching/` |
| Temporal Action Localization / 时序动作定位 | 10 | `dlhub/vision/temporal_action_localization/` |
| Gaze Estimation / 视线估计 | 10 | `dlhub/vision/gaze_estimation/` |
| Trajectory Prediction / 轨迹预测 | 10 | `dlhub/vision/trajectory_prediction/` |
| Scene Graph Generation / 场景图生成 | 10 | `dlhub/vision/scene_graph_generation/` |
| Camouflaged Object Detection / 伪装物体检测 | 10 | `dlhub/vision/camouflaged_object_detection/` |

## Research Directions / 研究方向（五）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Editing / 图像编辑 | 10 | `dlhub/vision/image_editing/` |
| Multi-focus Fusion / 多焦点图像融合 | 10 | `dlhub/vision/multi_focus_fusion/` |
| Online Handwriting Recognition / 联机手写汉字识别 | 10 | `dlhub/vision/online_handwriting_recognition/` |
| Lane Topology Estimation / 车道图估计 | 10 | `dlhub/vision/lane_topology_estimation/` |
| Remote Sensing Change Detection / 遥感变化检测 | 10 | `dlhub/vision/remote_sensing_change_detection/` |
| Cross-view Geo-localization / 跨视图地理定位 | 10 | `dlhub/vision/cross_view_geo_localization/` |
| Video Understanding / 视频理解 | 10 | `dlhub/vision/video_understanding/` |
| Video Enhancement / 视频增强 | 10 | `dlhub/vision/video_enhancement/` |
| Image Matching / 图像匹配 | 10 | `dlhub/vision/image_matching/` |
| Feature Matching / 特征匹配 | 10 | `dlhub/vision/feature_matching/` |

## Research Directions / 研究方向（六）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Low-light Enhancement / 低光增强 | 10 | `dlhub/vision/low_light_enhancement/` |
| Image Colorization / 图像上色 | 10 | `dlhub/vision/image_colorization/` |
| Referring Expression Comprehension / 指代表达理解 | 10 | `dlhub/vision/referring_expression_comprehension/` |
| Referring Expression Segmentation / 指代表达分割 | 10 | `dlhub/vision/referring_expression_segmentation/` |
| Open-vocabulary Segmentation / 开放词汇分割 | 10 | `dlhub/vision/open_vocabulary_segmentation/` |
| Video Temporal Grounding / 视频时序定位 | 10 | `dlhub/vision/video_temporal_grounding/` |
| Document Understanding / 文档理解 | 10 | `dlhub/vision/document_understanding/` |
| Shadow Removal / 阴影去除 | 10 | `dlhub/vision/shadow_removal/` |
| Reflection Removal / 反光去除 | 10 | `dlhub/vision/reflection_removal/` |
| Novel View Synthesis / 新视角合成 | 10 | `dlhub/vision/novel_view_synthesis/` |

## Research Directions / 研究方向（七）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Optical Flow / 光流估计 | 10 | `dlhub/vision/optical_flow/` |
| Person Search / 行人搜索 | 10 | `dlhub/vision/person_search/` |
| Human Parsing / 人体解析 | 10 | `dlhub/vision/human_parsing/` |
| Scene Text Spotting / 场景文本检测识别一体化 | 10 | `dlhub/vision/scene_text_spotting/` |
| Stereo Matching / 双目匹配 | 10 | `dlhub/vision/stereo_matching/` |
| Video Captioning / 视频描述 | 10 | `dlhub/vision/video_captioning/` |
| Video Question Answering / 视频问答 | 10 | `dlhub/vision/video_question_answering/` |
| Few-shot Recognition / 小样本识别 | 10 | `dlhub/vision/few_shot_recognition/` |
| Interactive Segmentation / 交互式分割 | 10 | `dlhub/vision/interactive_segmentation/` |
| Human Mesh Recovery / 人体网格恢复 | 10 | `dlhub/vision/human_mesh_recovery/` |

## Research Directions / 研究方向（八）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Quality Assessment / 图像质量评估 | 10 | `dlhub/vision/image_quality_assessment/` |
| Aesthetic Assessment / 美学评分 | 10 | `dlhub/vision/aesthetic_assessment/` |
| Video Quality Assessment / 视频质量评估 | 10 | `dlhub/vision/video_quality_assessment/` |
| Visual Dialog / 视觉对话 | 10 | `dlhub/vision/visual_dialog/` |
| Visual Entailment / 视觉蕴含 | 10 | `dlhub/vision/visual_entailment/` |
| Image Captioning / 图像描述 | 10 | `dlhub/vision/image_captioning/` |
| Phrase Grounding / 短语定位 | 10 | `dlhub/vision/phrase_grounding/` |
| Depth Completion / 深度补全 | 10 | `dlhub/vision/depth_completion/` |
| Surface Normal Estimation / 法线估计 | 10 | `dlhub/vision/surface_normal_estimation/` |
| Point Cloud Registration / 点云配准 | 10 | `dlhub/pointcloud/registration/` |

## Research Directions / 研究方向（九）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Quality Assessment / 图像质量评估 | 10 | `dlhub/vision/image_quality_assessment/` |
| Aesthetic Assessment / 美学评分 | 10 | `dlhub/vision/aesthetic_assessment/` |
| Video Quality Assessment / 视频质量评估 | 10 | `dlhub/vision/video_quality_assessment/` |
| Visual Dialog / 视觉对话 | 10 | `dlhub/vision/visual_dialog/` |
| Visual Entailment / 视觉蕴含 | 10 | `dlhub/vision/visual_entailment/` |
| Image Captioning / 图像描述 | 10 | `dlhub/vision/image_captioning/` |
| Phrase Grounding / 短语定位 | 10 | `dlhub/vision/phrase_grounding/` |
| Depth Completion / 深度补全 | 10 | `dlhub/vision/depth_completion/` |
| Surface Normal Estimation / 法线估计 | 10 | `dlhub/vision/surface_normal_estimation/` |
| Point Cloud Registration / 点云配准 | 10 | `dlhub/pointcloud/registration/` |

## Research Directions / 研究方向（十）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Edge Detection / 边缘检测 | 10 | `dlhub/vision/edge_detection/` |
| Line Segment Detection / 线段检测 | 10 | `dlhub/vision/line_segment_detection/` |
| Contour Detection / 轮廓检测 | 10 | `dlhub/vision/contour_detection/` |
| Defect Detection / 缺陷检测 | 10 | `dlhub/vision/defect_detection/` |
| Document Layout Analysis / 文档版面分析 | 10 | `dlhub/vision/document_layout_analysis/` |
| Table Structure Recognition / 表格结构识别 | 10 | `dlhub/vision/table_structure_recognition/` |
| Chart Understanding / 图表理解 | 10 | `dlhub/vision/chart_understanding/` |
| Fashion Compatibility / 时尚搭配预测 | 10 | `dlhub/vision/fashion_compatibility/` |
| Food Recognition / 食物识别 | 10 | `dlhub/vision/food_recognition/` |
| Symbol Recognition / 符号识别 | 10 | `dlhub/vision/symbol_recognition/` |

## Research Directions / 研究方向（十一）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Edge Detection / 边缘检测 | 10 | `dlhub/vision/edge_detection/` |
| Line Segment Detection / 线段检测 | 10 | `dlhub/vision/line_segment_detection/` |
| Contour Detection / 轮廓检测 | 10 | `dlhub/vision/contour_detection/` |
| Defect Detection / 缺陷检测 | 10 | `dlhub/vision/defect_detection/` |
| Document Layout Analysis / 文档版面分析 | 10 | `dlhub/vision/document_layout_analysis/` |
| Table Structure Recognition / 表格结构识别 | 10 | `dlhub/vision/table_structure_recognition/` |
| Chart Understanding / 图表理解 | 10 | `dlhub/vision/chart_understanding/` |
| Fashion Compatibility / 时尚搭配预测 | 10 | `dlhub/vision/fashion_compatibility/` |
| Food Recognition / 食物识别 | 10 | `dlhub/vision/food_recognition/` |
| Symbol Recognition / 符号识别 | 10 | `dlhub/vision/symbol_recognition/` |

## Research Directions / 研究方向（十二）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Visual Prompting / 视觉提示建模 | 10 | `dlhub/vision/visual_prompting/` |
| Visual Place Recognition / 视觉地点识别 | 10 | `dlhub/vision/visual_place_recognition/` |
| Video Prediction / 视频预测 | 10 | `dlhub/vision/video_prediction/` |
| Pose Tracking / 姿态跟踪 | 10 | `dlhub/vision/pose_tracking/` |
| Pedestrian Attribute Analysis / 行人属性分析 | 10 | `dlhub/vision/pedestrian_attribute_analysis/` |
| Object Counting / 目标计数 | 10 | `dlhub/vision/object_counting/` |
| Multimodal Fusion / 多模态融合 | 10 | `dlhub/vision/multimodal_fusion/` |
| Image Forensics / 图像取证 | 10 | `dlhub/vision/image_forensics/` |
| Graphical Document Parsing / 图形文档解析 | 10 | `dlhub/vision/graphical_document_parsing/` |
| Fine-Grained Retrieval / 细粒度检索 | 10 | `dlhub/vision/fine_grained_retrieval/` |

## Research Directions / 研究方向（十三）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Video Frame Interpolation / 视频插帧 | 10 | `dlhub/vision/video_frame_interpolation/` |
| Video Stabilization / 视频稳像 | 10 | `dlhub/vision/video_stabilization/` |
| Video Object Detection / 视频目标检测 | 10 | `dlhub/vision/video_object_detection/` |
| Document Dewarping / 文档矫正 | 10 | `dlhub/vision/document_dewarping/` |
| Layout Generation / 布局生成 | 10 | `dlhub/vision/layout_generation/` |
| Adversarial Robustness / 对抗鲁棒性 | 10 | `dlhub/vision/adversarial_robustness/` |
| Data Augmentation / 数据增广 | 10 | `dlhub/vision/data_augmentation/` |
| Image Synthesis / 图像合成 | 10 | `dlhub/vision/image_synthesis/` |
| Prompt Learning / 多模态 Prompt Learning | 10 | `dlhub/multimodal/prompt_learning/` |
| Gaussian Splatting / 3DGS | 10 | `dlhub/pointcloud/gaussian_splatting/` |

## Research Directions / 研究方向（十四）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Relighting / 图像重照明 | 10 | `dlhub/vision/image_relighting/` |
| Transparent Object Segmentation / 透明物体分割 | 10 | `dlhub/vision/transparent_object_segmentation/` |
| Video Matting / 视频抠像 | 10 | `dlhub/vision/video_matting/` |
| Event Camera Understanding / 事件相机理解 | 10 | `dlhub/vision/event_camera_understanding/` |
| Scene Flow / 场景流 | 10 | `dlhub/pointcloud/scene_flow/` |
| Point Cloud Completion / 点云补全 | 10 | `dlhub/pointcloud/pointcloud_completion/` |
| Audio-Visual Learning / 音视学习 | 10 | `dlhub/multimodal/audio_visual_learning/` |
| Multimodal Reasoning / 多模态推理 | 10 | `dlhub/multimodal/multimodal_reasoning/` |
| Video Diffusion / 视频扩散 | 10 | `dlhub/generative/video_diffusion/` |
| Text-to-3D / 文本生成三维 | 10 | `dlhub/generative/text_to_3d/` |

## Research Directions / 研究方向（十五）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Deraining / 图像去雨 | 10 | `dlhub/vision/image_deraining/` |
| Shadow Detection / 阴影检测 | 10 | `dlhub/vision/shadow_detection/` |
| Point Cloud Upsampling / 点云上采样 | 10 | `dlhub/pointcloud/pointcloud_upsampling/` |
| Shape Correspondence 3D / 三维形状对应 | 10 | `dlhub/pointcloud/shape_correspondence_3d/` |
| Open Vocabulary 3D / 开放词表三维 | 10 | `dlhub/pointcloud/open_vocabulary_3d/` |
| Image-Text Retrieval / 图文检索 | 10 | `dlhub/multimodal/image_text_retrieval/` |
| Vision-Language Navigation / 视觉语言导航 | 10 | `dlhub/multimodal/vision_language_navigation/` |
| Document VLM / 文档 VLM | 10 | `dlhub/multimodal/document_vlm/` |
| Image-to-Video / 图生视频 | 10 | `dlhub/generative/image_to_video/` |
| Image-to-3D / 图生三维 | 10 | `dlhub/generative/image_to_3d/` |

## Research Directions / 研究方向（十六）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Image Deweathering / 图像去天气 | 10 | `dlhub/vision/image_deweathering/` |
| Transparent Depth Estimation / 透明体深度估计 | 10 | `dlhub/vision/transparent_depth_estimation/` |
| Point Cloud Forecasting / 点云预测 | 10 | `dlhub/pointcloud/pointcloud_forecasting/` |
| Point Cloud Anomaly Detection / 点云异常检测 | 10 | `dlhub/pointcloud/pointcloud_anomaly_detection/` |
| Video-Text Retrieval / 视频文本检索 | 10 | `dlhub/multimodal/video_text_retrieval/` |
| Embodied Question Answering / 具身问答 | 10 | `dlhub/multimodal/embodied_question_answering/` |
| Audio-Text Understanding / 音频文本理解 | 10 | `dlhub/multimodal/audio_text_understanding/` |
| Text-to-Video / 文本生成视频 | 10 | `dlhub/generative/text_to_video/` |
| Video-to-Video / 视频生成视频 | 10 | `dlhub/generative/video_to_video/` |
| World Models / 世界模型 | 10 | `dlhub/generative/world_models/` |

## Research Directions / 研究方向（十七）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Pedestrian Detection / 行人检测 | 10 | `dlhub/vision/pedestrian_detection/` |
| Road Scene Understanding / 道路场景理解 | 10 | `dlhub/vision/road_scene_understanding/` |
| Salient Object Detection / 显著性目标检测 | 10 | `dlhub/vision/salient_object_detection/` |
| Face Attribute Recognition / 人脸属性识别 | 10 | `dlhub/vision/face_attribute_recognition/` |
| Face Occlusion Estimation / 人脸遮挡估计 | 10 | `dlhub/vision/face_occlusion_estimation/` |
| Deepfake Detection / 假脸检测 | 10 | `dlhub/vision/deepfake_detection/` |
| Face Verification / 人脸验证 | 10 | `dlhub/vision/face_verification/` |
| Face Identification / 人脸识别 | 10 | `dlhub/vision/face_identification/` |
| Face Retrieval / 人脸检索 | 10 | `dlhub/vision/face_retrieval/` |
| Face Pose Estimation / 人脸姿态估计 | 10 | `dlhub/vision/face_pose_estimation/` |

## Research Directions / 研究方向（十八）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Hand Pose Estimation / 手部姿态估计 | 10 | `dlhub/vision/hand_pose_estimation/` |
| Gesture Recognition / 手势识别 | 10 | `dlhub/vision/gesture_recognition/` |
| Finger Count Estimation / 手指计数估计 | 10 | `dlhub/vision/finger_count_estimation/` |
| Handedness Classification / 左右手分类 | 10 | `dlhub/vision/handedness_classification/` |
| Palm Orientation Estimation / 掌心朝向估计 | 10 | `dlhub/vision/palm_orientation_estimation/` |
| Sign Digit Classification / 手势数字分类 | 10 | `dlhub/vision/sign_digit_classification/` |
| Finger Spread Estimation / 手指张开度估计 | 10 | `dlhub/vision/finger_spread_estimation/` |
| Thumb Position Classification / 拇指位置分类 | 10 | `dlhub/vision/thumb_position_classification/` |
| Finger Curvature Estimation / 手指弯曲度估计 | 10 | `dlhub/vision/finger_curvature_estimation/` |
| Thumb Contact Classification / 拇指接触分类 | 10 | `dlhub/vision/thumb_contact_classification/` |

## Research Directions / 研究方向（十九）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Anchor-Free Detection / Anchor-Free 检测 | 10 | `dlhub/vision/anchor_free_detection/` |
| Keypoint Regression / 关键点回归 | 10 | `dlhub/vision/keypoint_regression/` |
| UNet Segmentation / UNet 分割 | 10 | `dlhub/vision/unet_segmentation/` |
| YOLACT Instance Segmentation / YOLACT 实例分割 | 10 | `dlhub/vision/yolact_instance_segmentation/` |
| Neural Style Transfer / 神经风格迁移 | 10 | `dlhub/vision/neural_style_transfer/` |
| Image Translation / 图像翻译 | 10 | `dlhub/vision/image_translation/` |
| Monocular Depth Estimation / 单目深度估计 | 10 | `dlhub/vision/monocular_depth_estimation/` |
| Salient Object Detection Boxes / 显著性目标框检测 | 10 | `dlhub/vision/salient_object_detection_boxes/` |
| Face Landmark Detection / 人脸关键点检测 | 10 | `dlhub/vision/face_landmark_detection/` |
| Face Liveness Detection / 人脸活体检测 | 10 | `dlhub/vision/face_liveness_detection/` |

## Research Directions / 研究方向（二十）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Open Vocabulary Detection / 开放词表检测 | 10 | `dlhub/vision/open_vocabulary_detection/` |
| Few-Shot Segmentation / 小样本分割 | 10 | `dlhub/vision/few_shot_segmentation/` |
| Object Proposal Generation / 目标候选框生成 | 10 | `dlhub/vision/object_proposal_generation/` |
| Image Moire Removal / 图像去摩尔纹 | 10 | `dlhub/vision/image_moire_removal/` |
| Raindrop Removal / 雨滴去除 | 10 | `dlhub/vision/raindrop_removal/` |
| Compression Artifact Reduction / 压缩伪影去除 | 10 | `dlhub/vision/compression_artifact_reduction/` |
| Document Binarization / 文档二值化 | 10 | `dlhub/vision/document_binarization/` |
| Crowd Localization / 人群定位 | 10 | `dlhub/vision/crowd_localization/` |
| Homography Estimation / 单应性估计 | 10 | `dlhub/vision/homography_estimation/` |
| Camera Pose Estimation / 相机位姿估计 | 10 | `dlhub/vision/camera_pose_estimation/` |

## Research Directions / 研究方向（二十一）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Infrared Small Target Detection / 红外小目标检测 | 10 | `dlhub/vision/infrared_small_target_detection/` |
| Blur Detection / 模糊检测 | 10 | `dlhub/vision/blur_detection/` |
| Camera Calibration / 相机标定 | 10 | `dlhub/vision/camera_calibration/` |
| Vanishing Point Detection / 消失点检测 | 10 | `dlhub/vision/vanishing_point_detection/` |
| Image Outpainting / 图像外延生成 | 10 | `dlhub/vision/image_outpainting/` |
| Underwater Image Enhancement / 水下图像增强 | 10 | `dlhub/vision/underwater_image_enhancement/` |
| Gaze Following / 视线跟随 | 10 | `dlhub/vision/gaze_following/` |
| Object Discovery / 目标发现 | 10 | `dlhub/vision/object_discovery/` |
| Motion Segmentation / 运动分割 | 10 | `dlhub/vision/motion_segmentation/` |
| Salient Instance Segmentation / 显著性实例分割 | 10 | `dlhub/vision/salient_instance_segmentation/` |

## Research Directions / 研究方向（二十二）

| 方向 | 当前家族数 | 包路径 |
|------|-----------|--------|
| Mirror Segmentation / 镜面分割 | 10 | `dlhub/vision/mirror_segmentation/` |
| Hand Segmentation / 手部分割 | 10 | `dlhub/vision/hand_segmentation/` |
| Iris Segmentation / 虹膜分割 | 10 | `dlhub/vision/iris_segmentation/` |
| Pupil Detection / 瞳孔检测 | 10 | `dlhub/vision/pupil_detection/` |
| Crack Detection / 裂缝检测 | 10 | `dlhub/vision/crack_detection/` |
| Glare Detection / 眩光检测 | 10 | `dlhub/vision/glare_detection/` |
| Lens Flare Removal / 镜头光斑去除 | 10 | `dlhub/vision/lens_flare_removal/` |
| Illumination Estimation / 光照估计 | 10 | `dlhub/vision/illumination_estimation/` |
| Exposure Correction / 曝光校正 | 10 | `dlhub/vision/exposure_correction/` |
| Reflection Detection / 反射检测 | 10 | `dlhub/vision/reflection_detection/` |
