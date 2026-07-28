# Fine-Grained Visual Recognition (FGVC)

本目录提供**细粒度视觉识别**（Fine-Grained Visual Recognition, FGVC）的 compact-first / pure-torch 实现：

- 不依赖预训练权重，不触发任何下载
- 统一为 `build_*_fgvc_classifier(...)` 工厂函数
- 每个算法族提供 `*_tiny / *_small / *_base` 三个 variants
- 模型输出统一为 `dict`，且必须包含 `logits`

配套可发现性工具：

```bash
python scripts/fine_grained_recognition_zoo.py --list
python scripts/fine_grained_recognition_zoo.py --search transfg --limit 20
python scripts/fine_grained_recognition_zoo.py --smoke dlfgvc:transfg_tiny
```

## Timeline (按年份的代表性里程碑)

注意：这里的实现是**教学/验证友好**的“结构化 compact 版本”，目的是让你能在本仓库里快速跑通 forward/backward、做对比实验，
而不是逐行复现论文的完整训练 recipe 或性能。

| 年份 | 代表方法 | 本仓库 family / 例子 |
|---|---|---|
| 2015 | Bilinear CNN (B-CNN) | `bilinear_cnn` / `dlfgvc:bilinear_cnn_tiny` |
| 2016 | Compact Bilinear Pooling | `compact_bilinear` / `dlfgvc:compact_bilinear_tiny` |
| 2017 | RA-CNN (Recurrent Attention) | `racnn` / `dlfgvc:racnn_tiny` |
| 2018 | NTS-Net (proposal + zoom-in) | `nts_net` / `dlfgvc:nts_net_tiny` |
| 2019 | DCL (Deep Complementary Learning) | `dcl` / `dlfgvc:dcl_tiny` |
| 2020 | PMG (progressive multi-granularity) | `pmg` / `dlfgvc:pmg_tiny` |
| 2021 | TransFG (token selection) | `transfg` / `dlfgvc:transfg_tiny` |
| 2022 | VPT (Visual Prompt Tuning) | `vpt` / `dlfgvc:vpt_tiny` |
| 2023 | SM-ViT (salient mask guided ViT) | `sm_vit` / `dlfgvc:sm_vit_tiny` |
| 2024 | LDH-ViT (local concealment + selection) | `ldh_vit` / `dlfgvc:ldh_vit_tiny` |
| 2025 | Prompt-CAM (interpretable prompt attention map) | `prompt_cam` / `dlfgvc:prompt_cam_tiny` |
| 2025 | Finer-CAM (spotting difference for explanation) | `finer_cam` / `dlfgvc:finer_cam_tiny` |
| 2025 | XR-VLM (multi-part prompts + class interaction) | `xr_vlm` / `dlfgvc:xr_vlm_tiny` |
| 2025 | FG-CLIP (CLIP-style alignment for FGVC) | `fg_clip` / `dlfgvc:fg_clip_tiny` |
| 2026 | Zooming without Zooming (region-to-image distillation) | `r2i_distill` / `dlfgvc:r2i_distill_tiny` |
| 2026 | ImgCoT (compact visual CoT tokens) | `img_cot` / `dlfgvc:img_cot_tiny` |
| 2026 | ReFine-RFT (reasoning length regulation) | `refine_rft` / `dlfgvc:refine_rft_tiny` |
| 2026 | IIR-VLM (instance-level expert fusion) | `iir_vlm` / `dlfgvc:iir_vlm_tiny` |
| 2026 | Fine-R1 (MLLM CoT-style reasoning for FGVC) | `fine_r1` / `dlfgvc:fine_r1_tiny` |

额外补充（不严格按年份）：`gem_pooling` 提供 GeM pooling head 的一个强基线形态。

## Archive (FGVC 全部 family 按年份归档)

这个归档表覆盖本目录当前所有 family（与 `python scripts/fine_grained_recognition_zoo.py --list` 保持一致）。
也可以用 CLI 直接查看：

```bash
python scripts/fine_grained_recognition_zoo.py --timeline
```

说明：

- 年份为 best-effort（以代表性论文/最早常用版本为主），与预印本/期刊最终版本可能存在 1 年左右差异
- 本仓库实现是 “compact 结构复刻”，用于学习、对比与快速验证；不承诺复现原论文完整训练 recipe
- 归档元数据来源：`dlhub/vision/fine_grained_recognition/_timeline.py`

| 年份 | Group | family | 方法 (简写) | 例子 |
|---|---|---|---|---|
| 2014 | `part` | `part_rcnn` | Part-based R-CNN (parts + crop) | `dlfgvc:part_rcnn_tiny` |
| 2015 | `bilinear` | `bilinear_cnn` | Bilinear CNN (B-CNN) | `dlfgvc:bilinear_cnn_tiny` |
| 2016 | `bilinear` | `compact_bilinear` | Compact Bilinear Pooling | `dlfgvc:compact_bilinear_tiny` |
| 2016 | `part` | `part_stacked_cnn` | Part-Stacked CNN | `dlfgvc:part_stacked_cnn_tiny` |
| 2017 | `bilinear` | `kernel_pooling` | Kernel Pooling (bilinear variant) | `dlfgvc:kernel_pooling_tiny` |
| 2017 | `bilinear` | `lowrank_bilinear` | Low-rank / Factorized Bilinear Pooling | `dlfgvc:lowrank_bilinear_tiny` |
| 2017 | `part` | `ma_cnn` | MA-CNN (multi-attention) | `dlfgvc:ma_cnn_tiny` |
| 2017 | `part` | `pa_cnn` | PA-CNN (part-aligned / part-attention CNN) | `dlfgvc:pa_cnn_tiny` |
| 2017 | `part` | `racnn` | RA-CNN (recurrent attention) | `dlfgvc:racnn_tiny` |
| 2017 | `relation` | `ga_cnn` | GA-CNN (granularity-aware) | `dlfgvc:ga_cnn_tiny` |
| 2017 | `relation` | `interp_parts` | Interpretable Part Modeling | `dlfgvc:interp_parts_tiny` |
| 2018 | `bilinear` | `gem_pooling` | GeM pooling head (strong baseline) | `dlfgvc:gem_pooling_tiny` |
| 2018 | `bilinear` | `hierarchical_bilinear` | Hierarchical Bilinear Pooling | `dlfgvc:hierarchical_bilinear_tiny` |
| 2018 | `bilinear` | `isqrt_cov` | iSQRT-COV (iterative matrix sqrt for covariance pooling) | `dlfgvc:isqrt_cov_tiny` |
| 2018 | `bilinear` | `mpn_cov` | MPN-COV (matrix power normalized covariance) | `dlfgvc:mpn_cov_tiny` |
| 2018 | `bilinear` | `ws_ban` | WS-BAN (weakly supervised bilinear attention) | `dlfgvc:ws_ban_tiny` |
| 2018 | `part` | `nts_net` | NTS-Net (proposal + zoom-in) | `dlfgvc:nts_net_tiny` |
| 2018 | `relation` | `hse` | HSE (hierarchical semantic embedding) | `dlfgvc:hse_tiny` |
| 2018 | `relation` | `osme_mamc` | OSME + MAMC (multi-attention + constraint) | `dlfgvc:osme_mamc_tiny` |
| 2019 | `part` | `dfl_cnn` | DFL-CNN (discriminative filter learning) | `dlfgvc:dfl_cnn_tiny` |
| 2019 | `part` | `mge_cnn` | MGE-CNN (multi-granularity ensemble) | `dlfgvc:mge_cnn_tiny` |
| 2019 | `part` | `partnet` | PartNet (part mining / part discovery) | `dlfgvc:partnet_tiny` |
| 2019 | `part` | `s3n` | S3N (snapshot + zoom) | `dlfgvc:s3n_tiny` |
| 2019 | `part` | `tasn` | TASN (trilinear attention sampling) | `dlfgvc:tasn_tiny` |
| 2019 | `relation` | `api_net` | API-Net (attentive pairwise interaction) | `dlfgvc:api_net_tiny` |
| 2019 | `relation` | `crossx` | CrossX (cross-region interaction) | `dlfgvc:crossx_tiny` |
| 2019 | `relation` | `dcl` | DCL (deep complementary learning) | `dlfgvc:dcl_tiny` |
| 2019 | `relation` | `proto_pnet` | ProtoPNet (prototype-based interpretability) | `dlfgvc:proto_pnet_tiny` |
| 2019 | `relation` | `ws_dan` | WS-DAN (weakly supervised data augmentation) | `dlfgvc:ws_dan_tiny` |
| 2020 | `part` | `pmg` | PMG (progressive multi-granularity) | `dlfgvc:pmg_tiny` |
| 2020 | `relation` | `region_grouping` | Region Grouping (grouped regions/parts) | `dlfgvc:region_grouping_tiny` |
| 2020 | `transformer` | `pca_net` | PCA-Net (co-attention style transformer) | `dlfgvc:pca_net_tiny` |
| 2021 | `transformer` | `cvl` | CVL (vision-language token fusion) | `dlfgvc:cvl_tiny` |
| 2021 | `transformer` | `ffvt` | FFVT (fine-grained feature fusion ViT) | `dlfgvc:ffvt_tiny` |
| 2021 | `transformer` | `pedtrans` | PedTrans (pose/metadata guided transformer) | `dlfgvc:pedtrans_tiny` |
| 2021 | `transformer` | `pim` | PIM (plug-in / part interaction module) | `dlfgvc:pim_tiny` |
| 2021 | `transformer` | `sim_trans` | Sim-Trans (similarity-driven transformer) | `dlfgvc:sim_trans_tiny` |
| 2021 | `transformer` | `transfg` | TransFG (token selection for FGVC) | `dlfgvc:transfg_tiny` |
| 2022 | `transformer` | `aftrans` | AFTrans (attention fusion transformer) | `dlfgvc:aftrans_tiny` |
| 2022 | `transformer` | `metaformer_fgvc` | MetaFormer (backbone-style transformer family) | `dlfgvc:metaformer_fgvc_tiny` |
| 2022 | `transformer` | `vit_fod` | ViT-FOD (feature/object difference token) | `dlfgvc:vit_fod_tiny` |
| 2022 | `transformer` | `vpt` | VPT (Visual Prompt Tuning) | `dlfgvc:vpt_tiny` |
| 2023 | `transformer` | `sm_vit` | SM-ViT (salient mask guided ViT) | `dlfgvc:sm_vit_tiny` |
| 2024 | `transformer` | `ldh_vit` | LDH-ViT (local concealment + selection) | `dlfgvc:ldh_vit_tiny` |
| 2025 | `transformer` | `fg_clip` | FG-CLIP (CLIP-style visual-text alignment) | `dlfgvc:fg_clip_tiny` |
| 2025 | `transformer` | `finer_cam` | Finer-CAM (difference spotting for explanation) | `dlfgvc:finer_cam_tiny` |
| 2025 | `transformer` | `prompt_cam` | Prompt-CAM (interpretable prompt attention map) | `dlfgvc:prompt_cam_tiny` |
| 2025 | `transformer` | `xr_vlm` | XR-VLM (multi-part prompts + cross-relationship modeling) | `dlfgvc:xr_vlm_tiny` |
| 2026 | `transformer` | `fine_r1` | Fine-R1 (CoT-style reasoning tokens for FGVC) | `dlfgvc:fine_r1_tiny` |
| 2026 | `transformer` | `r2i_distill` | Zooming without Zooming (region-to-image distillation) | `dlfgvc:r2i_distill_tiny` |
| 2026 | `transformer` | `iir_vlm` | IIR-VLM (instance-level expert fusion for VLM) | `dlfgvc:iir_vlm_tiny` |
| 2026 | `transformer` | `img_cot` | ImgCoT (compact visual CoT tokens) | `dlfgvc:img_cot_tiny` |
| 2026 | `transformer` | `refine_rft` | ReFine-RFT (reasoning length regulation / cost of thinking) | `dlfgvc:refine_rft_tiny` |
