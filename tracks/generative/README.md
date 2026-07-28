# Generative 轨（生成模型）

目标：用 **compact-first** 的方式建立生成建模直觉，并且保持与仓库其它轨道一致的训练/输出规范：

- 能跑通（CPU 也能快速冒烟）
- 能看懂（代码短、注释少但结构清晰）
- 能改动（每节课有练习建议）
- 能验收（输出目录统一，`scripts/smoke_check.py` 覆盖）

> 说明：本轨道默认不强依赖 `torchvision`。如果你选择 `--dataset mnist` 才需要安装 `torchvision`。

## Lessons

- `lesson_01_vae_mnist/`：Vanilla VAE（最小实现，支持 `--dataset fake`）
- `lesson_02_gan_mnist/`：Vanilla GAN（最小实现，MLP 结构，支持 `--dataset fake`）
- `lesson_03_compact_diffusion_mnist/`：Compact DDPM（像素空间噪声预测 + 线性噪声日程）
- `lesson_04_compact_latent_diffusion/`：Compact Latent Diffusion（自编码器潜空间 + 潜变量去噪）
- `lesson_05_compact_consistency_model/`：Compact Consistency Model（一步一致性映射 + 蒸馏式采样）
- `lesson_06_compact_flow_matching/`：Compact Flow Matching（向量场回归 + 连续时间输运）
- `lesson_07_compact_rectified_flow/`：Compact Rectified Flow（直线路径输运 + 速度场回归）
- `lesson_08_compact_diffusion_transformer/`：Compact Diffusion Transformer（patch token 去噪 + DiT 风格骨干）
- `lesson_09_compact_conditional_gan/`：Compact Conditional GAN（标签条件生成 + 条件判别器）
- `lesson_10_compact_diffusion_image_editing/`：Compact Diffusion Image Editing（源图条件 + 编辑掩码 + 噪声预测）
- `lesson_11_compact_controlnet/`：Compact ControlNet（结构提示分支 + 条件残差注入 + 受控去噪）
- `lesson_12_compact_layout_to_image/`：Compact Layout-to-Image（布局框编码 + 对象组合渲染 + 条件生成）
- `lesson_13_compact_text_to_image_diffusion/`：Compact Text-to-Image Diffusion（文本提示条件 + 合成场景扩散生成）
- `lesson_14_compact_diffusion_inpainting/`：Compact Diffusion Inpainting（掩码区域修复 + 上下文条件去噪）
- `lesson_15_compact_diffusion_super_resolution/`：Compact Diffusion Super-Resolution（低分辨率条件去噪 + 上采样细节恢复）
- `lesson_16_compact_diffusion_deblurring/`：Compact Diffusion Deblurring（模糊图条件去噪 + 清晰图重建）
- `lesson_17_compact_diffusion_denoising/`：Compact Diffusion Denoising（噪声图条件去噪 + 配对干净图恢复）
- `lesson_18_compact_diffusion_deraining/`：Compact Diffusion Deraining（雨条纹条件去噪 + 配对去雨恢复）
- `lesson_19_compact_diffusion_dehazing/`：Compact Diffusion Dehazing（雾化图条件去噪 + 配对清晰图恢复）
- `lesson_20_compact_diffusion_reflection_removal/`：Compact Diffusion Reflection Removal（反光图条件去噪 + 透射内容恢复）
- `lesson_21_compact_diffusion_image_fusion/`：Compact Diffusion Image Fusion（多源互补观测条件去噪 + 融合重建）
- `lesson_22_compact_diffusion_style_transfer/`：Compact Diffusion Style Transfer（内容/风格双条件去噪 + 纹理迁移重建）
- `lesson_23_compact_diffusion_multi_focus_fusion/`：Compact Diffusion Multi-Focus Fusion（双焦平面条件去噪 + 清晰区域互补融合）
- `lesson_24_compact_diffusion_image_synthesis/`：Compact Diffusion Image Synthesis（结构条件去噪 + 简单场景合成）
- `lesson_25_compact_diffusion_compositional_generation/`：Compact Diffusion Compositional Generation（结构/纹理双条件组合 + 扩散式图像合成）
- `lesson_26_compact_diffusion_image_variation/`：Compact Diffusion Image Variation（源图条件扩散重采样 + 合成图像变体生成）
- `lesson_27_compact_diffusion_reference_guided_generation/`：Compact Diffusion Reference-Guided Generation（reference/condition 双条件去噪 + 外观参照引导）
- `lesson_28_compact_diffusion_subject_driven_generation/`：Compact Diffusion Subject-Driven Generation（主体条件保持 + guidance 控制 + 扩散采样）
- `lesson_29_compact_diffusion_multi_reference_generation/`：Compact Diffusion Multi-Reference Generation（双 reference + condition 联合去噪 + 多条件采样）
- `lesson_30_compact_diffusion_identity_preserving_editing/`：Compact Diffusion Identity-Preserving Editing（identity/source 双条件编辑 + 身份保持采样）
- `lesson_31_compact_diffusion_reference_editing/`：Compact Diffusion Reference Editing（source/reference 双条件编辑 + 外观借用采样）
- `lesson_32_compact_diffusion_layout_preserving_editing/`：Compact Diffusion Layout-Preserving Editing（layout/edit 双条件编辑 + 全局结构保持）
- `lesson_33_compact_diffusion_masked_reference_editing/`：Compact Diffusion Masked Reference Editing（source/reference/mask 三条件编辑 + 局部区域控制）
- `lesson_34_compact_diffusion_layout_reference_fusion/`：Compact Diffusion Layout-Reference Fusion（layout/reference 双条件融合 + 结构纹理解耦）
- `lesson_35_compact_diffusion_box_mask_editing/`：Compact Diffusion Box-Mask Editing（source/box-mask 双条件编辑 + 矩形局部重写）
- `lesson_36_compact_diffusion_layout_subject_fusion/`：Compact Diffusion Layout-Subject Fusion（layout/subject 双条件融合 + 结构主体解耦）
- `lesson_37_compact_diffusion_polygon_mask_editing/`：Compact Diffusion Polygon-Mask Editing（source/polygon-mask 双条件编辑 + 多边形局部重写）
- `lesson_38_compact_diffusion_layout_attribute_fusion/`：Compact Diffusion Layout-Attribute Fusion（layout/attribute 双条件融合 + 结构属性解耦）
- `lesson_39_compact_diffusion_scribble_mask_editing/`：Compact Diffusion Scribble-Mask Editing（source/scribble-mask 双条件编辑 + 稀疏涂鸦局部重写）
- `lesson_40_compact_diffusion_layout_style_fusion/`：Compact Diffusion Layout-Style Fusion（layout/style 双条件融合 + 结构风格解耦）
- `lesson_41_compact_diffusion_stroke_mask_editing/`：Compact Diffusion Stroke-Mask Editing（source/stroke-mask 双条件编辑 + 画笔轨迹局部重写）
- `lesson_42_compact_diffusion_layout_palette_fusion/`：Compact Diffusion Layout-Palette Fusion（layout/palette 双条件融合 + 结构配色解耦）
- `lesson_43_compact_diffusion_path_mask_editing/`：Compact Diffusion Path-Mask Editing（source/path-mask 双条件编辑 + 路径轨迹局部重写）
- `lesson_44_compact_diffusion_layout_lighting_fusion/`：Compact Diffusion Layout-Lighting Fusion（layout/lighting 双条件融合 + 结构光照解耦）
- `lesson_45_compact_video_diffusion/`：Compact Video Diffusion（多帧 clip 条件去噪 + 时间一致性建模）
- `lesson_46_compact_image_to_video_diffusion/`：Compact Image-to-Video Diffusion（源图条件短视频生成 + motion-conditioned 去噪）
- `lesson_47_compact_text_to_3d/`：Compact Text-to-3D（文本条件三维表示生成 + triplane/density/mesh token 监督）
- `lesson_48_compact_image_to_3d/`：Compact Image-to-3D（单图条件三维提升 + density/mesh token 重建）
- `lesson_49_compact_text_to_video/`：Compact Text-to-Video（文本条件短视频生成 + prompt-conditioned motion 建模）
- `lesson_50_compact_video_to_video/`：Compact Video-to-Video（源视频条件变换 + residual/mix video translation）
- `lesson_51_compact_world_models/`：Compact World Models（潜在动力学建模 + 短轨迹 rollout 重建）
