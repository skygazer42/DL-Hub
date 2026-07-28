# Action Recognition (Video + Skeleton)

本目录提供**行为识别 / 动作识别**（Action Recognition）的 compact-first / pure-torch 实现，覆盖两种常见模态：

- **Video**：输入视频张量 `x` 形状为 `(B, C, T, H, W)`
- **Skeleton**：输入骨架张量 `x` 形状为 `(B, C, T, V)`，其中 `V` 为关节点数

设计目标与约束：

- 不依赖预训练权重，不触发任何下载
- 一文件一算法族，variants 统一 `_tiny/_small/_base`
- 统一 factory：`build_*_video_classifier(...)` / `build_*_skeleton_classifier(...)`
- forward 输出 logits（Tensor），方便接入通用的 `fit_classifier(...)`

配套可发现性工具：

```bash
python scripts/action_recognition_zoo.py --list
python scripts/action_recognition_zoo.py --search stgcn
python scripts/action_recognition_zoo.py --timeline
python scripts/action_recognition_zoo.py --smoke dlactv:c3d_tiny
python scripts/action_recognition_zoo.py --smoke dlacts:stgcn_tiny
```

## Timeline (按年份的代表性里程碑)

注意：这里的实现是教学/验证友好的“结构化 compact 版本”，目的是让你能在本仓库里快速跑通 forward/backward、对比结构差异，
而不是逐行复现论文的完整训练 recipe 或 SOTA 指标。

| 年份 | 模态 | 代表方法 | 本仓库 family / 例子 |
|---|---|---|---|
| 2014 | Video | Two-Stream CNN (RGB + motion) | `two_stream` / `dlactv:two_stream_tiny` |
| 2015 | Video | C3D (3D CNN) | `c3d` / `dlactv:c3d_tiny` |
| 2016 | Video | TSN (segment consensus) | `tsn` / `dlactv:tsn_tiny` |
| 2017 | Video | I3D (inflated 3D conv) | `i3d` / `dlactv:i3d_tiny` |
| 2018 | Skeleton | ST-GCN | `stgcn` / `dlacts:stgcn_tiny` |
| 2018 | Video | R(2+1)D (factorized 3D conv) | `r2plus1d` / `dlactv:r2plus1d_tiny` |
| 2018 | Video | Non-local block (space-time self-attention) | `non_local` / `dlactv:non_local_tiny` |
| 2019 | Video | TSM (temporal shift) | `tsm` / `dlactv:tsm_tiny` |
| 2019 | Video | SlowFast | `slowfast` / `dlactv:slowfast_tiny` |
| 2019 | Skeleton | 2S-AGCN | `agcn` / `dlacts:agcn_tiny` |
| 2020 | Video | X3D (efficient 3D conv) | `x3d` / `dlactv:x3d_tiny` |
| 2020 | Skeleton | Shift-GCN | `shift_gcn` / `dlacts:shift_gcn_tiny` |
| 2020 | Skeleton | MS-G3D (multi-hop graph conv) | `ms_g3d` / `dlacts:ms_g3d_tiny` |
| 2021 | Video | TimeSformer | `timesformer` / `dlactv:timesformer_tiny` |
| 2021 | Video | ViViT (factorized video transformer) | `vivit` / `dlactv:vivit_tiny` |
| 2021 | Skeleton | CTR-GCN | `ctr_gcn` / `dlacts:ctr_gcn_tiny` |
| 2021 | Skeleton | PoseFormer | `poseformer` / `dlacts:poseformer_tiny` |
| 2021 | Skeleton | ST-Transformer (factorized attention) | `sttr` / `dlacts:sttr_tiny` |
| 2022 | Video | VideoMAE (tubelet ViT) | `videomae` / `dlactv:videomae_tiny` |
| 2022 | Skeleton | MotionBERT (masked motion modeling) | `motionbert` / `dlacts:motionbert_tiny` |
| 2024 | Video | VideoMamba (SSM/Mamba-style mixer) | `videomamba` / `dlactv:videomamba_tiny` |
| 2025 | Video | VideoRNN (CNN + GRU, efficient temporal modeling) | `videornn` / `dlactv:videornn_tiny` |

## Archive (全部 family 按年份归档)

这个归档表覆盖本目录当前所有 family（与 `python scripts/action_recognition_zoo.py --list` 保持一致）。

说明：

- 年份为 best-effort（以代表性论文/最早常用版本为主），与预印本/期刊最终版本可能存在 1 年左右差异
- 本仓库实现是“compact 结构复刻”，用于学习、对比与快速验证；不承诺复现原论文完整训练 recipe
- 归档元数据来源：`dlhub/vision/action_recognition/_timeline.py`

用 CLI 也可查看：

```bash
python scripts/action_recognition_zoo.py --timeline
```

| 年份 | 模态 | family | 方法 (简写) | 例子 |
|---|---|---|---|---|
| 2014 | `video` | `two_stream` | Two-Stream CNN (RGB + motion stream, compact) | `dlactv:two_stream_tiny` |
| 2015 | `video` | `c3d` | C3D (3D CNN baseline) | `dlactv:c3d_tiny` |
| 2016 | `video` | `tsn` | TSN (segment sampling + consensus) | `dlactv:tsn_tiny` |
| 2017 | `video` | `i3d` | I3D (inflated 3D conv, compact) | `dlactv:i3d_tiny` |
| 2018 | `skeleton` | `stgcn` | ST-GCN (spatio-temporal graph conv) | `dlacts:stgcn_tiny` |
| 2018 | `video` | `non_local` | Non-local block (space-time self-attention, compact) | `dlactv:non_local_tiny` |
| 2018 | `video` | `r2plus1d` | R(2+1)D (factorized 3D conv, compact) | `dlactv:r2plus1d_tiny` |
| 2019 | `skeleton` | `agcn` | 2S-AGCN (adaptive graph conv, compact) | `dlacts:agcn_tiny` |
| 2019 | `video` | `slowfast` | SlowFast (dual-pathway) | `dlactv:slowfast_tiny` |
| 2019 | `video` | `tsm` | TSM (temporal shift module) | `dlactv:tsm_tiny` |
| 2020 | `skeleton` | `ms_g3d` | MS-G3D (multi-hop graph conv, compact) | `dlacts:ms_g3d_tiny` |
| 2020 | `skeleton` | `shift_gcn` | Shift-GCN (shift operator on joints/time, compact) | `dlacts:shift_gcn_tiny` |
| 2020 | `video` | `x3d` | X3D (efficient 3D conv, compact) | `dlactv:x3d_tiny` |
| 2021 | `skeleton` | `ctr_gcn` | CTR-GCN (dynamic topology refinement, compact) | `dlacts:ctr_gcn_tiny` |
| 2021 | `skeleton` | `poseformer` | PoseFormer (transformer over joints/time, compact) | `dlacts:poseformer_tiny` |
| 2021 | `skeleton` | `sttr` | ST-Transformer (factorized spatial+temporal attention, compact) | `dlacts:sttr_tiny` |
| 2021 | `video` | `timesformer` | TimeSformer (space-time attention) | `dlactv:timesformer_tiny` |
| 2021 | `video` | `vivit` | ViViT (factorized video transformer, compact) | `dlactv:vivit_tiny` |
| 2022 | `skeleton` | `motionbert` | MotionBERT (masked motion modeling, compact) | `dlacts:motionbert_tiny` |
| 2022 | `video` | `videomae` | VideoMAE (tubelet ViT, compact) | `dlactv:videomae_tiny` |
| 2024 | `video` | `videomamba` | VideoMamba (SSM/Mamba-style mixer, compact) | `dlactv:videomamba_tiny` |
| 2025 | `video` | `videornn` | VideoRNN (CNN+GRU, efficient temporal modeling, compact) | `dlactv:videornn_tiny` |
