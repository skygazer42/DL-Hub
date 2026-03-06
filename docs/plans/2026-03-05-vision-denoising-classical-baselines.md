# Vision Denoising: Classical Baselines (12 算法族)

**Goal:** 在 `dlhub/vision/denoising/` 增加一批「CPU 友好、纯 torch、toy-first」的传统/非学习型去噪基线，用于快速对照与课程实验。

**New algorithm families (one file per family):**

- `median_filter.py` — 中值滤波（对椒盐/脉冲噪声很有用）
- `wiener_filter.py` — 局部 Wiener（基于局部均值/方差）
- `guided_filter.py` — 导向滤波（边缘保持平滑）
- `bilateral_filter.py` — 双边滤波（空间+强度权重）
- `non_local_means.py` — 非局部均值（NLM，简化版）
- `total_variation.py` — TV/ROF 去噪（Chambolle 迭代）
- `anisotropic_diffusion.py` — 各向异性扩散（Perona–Malik）
- `wavelet_shrinkage.py` — Haar 小波阈值（软阈值收缩）
- `anscombe_wiener.py` — Anscombe 变换 + Wiener（Poisson-ish 噪声基线）
- `lee_filter.py` — Lee speckle filter（乘性噪声 / speckle）
- `kuan_filter.py` — Kuan speckle filter（乘性噪声 / speckle）
- `stripe_remover.py` — 条纹/带状噪声去除（stripe / banding）

**Integration:**
- 统一 builder 风格：每个文件都提供 `_VARIANTS` + `build_*_denoiser(...)` + `__main__` smoke。
- Lesson 10 训练脚本支持「非可训练 baseline」：如果模型没有可训练参数则只做 `val` 评估并退出（避免空参数 optimizer 报错）。

**Verification:**
- `pytest -q`（新增 `tests/test_tracks_vision_denoising.py` baseline forward smoke）
