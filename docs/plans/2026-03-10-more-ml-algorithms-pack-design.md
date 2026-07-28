# 更多经典 ML 算法（NumPy 手写）设计说明

**日期**：2026-03-10  
**范围**：DL-Hub / `ml_algorithms/python/`（NumPy 经典算法库）

## 背景

仓库已包含一套以 NumPy 实现的经典机器学习算法集合（`ml_algorithms/python/`），并通过 pytest 做基础验收。

本需求希望在保持仓库风格一致（compact-first、可读、可测）的前提下，新增一组“更多算法”，并用“**8 个分支 → 合并到 1 个整合分支**”的工作流完成。

## 目标（Goals）

1. 新增 **8 个**经典 ML 算法（NumPy 实现），每个算法一个新模块文件，便于并行/低冲突开发。
2. 每个算法都提供最小可用 API（与现有代码风格一致）：
   - `fit(...) -> self`
   - `predict(...)` 或 `transform(...)` / `score_samples(...)` 等核心接口
3. 为新增算法提供 **稳定、快速** 的单元测试（`pytest -q` 全量通过）。
4. 更新 `ml_algorithms/python/README.md` 的算法列表，保持文档同步。

## 非目标（Non-goals）

- 不追求与 sklearn 完全一致的超参数/边界行为（不做 1:1 复刻）。
- 不做高性能优化（不引入 Cython/numba/并行加速）。
- 不引入新第三方依赖（保持 `numpy` 即可运行）。

## 新增算法清单（8 个）

优先选择“可用、可测、独立文件、对现有集合有补充”的算法：

1. **Lasso Regression**（坐标下降 + soft-threshold）
2. **Elastic Net Regression**（L1+L2，坐标下降）
3. **Kernel Ridge Regression**（dual 解，支持 linear/RBF kernel）
4. **Gaussian Process Regressor (RBF)**（Cholesky 解，支持 `predict(return_std=True)`）
5. **Kernel PCA**（kernel centering + eigendecomposition，支持 linear/RBF）
6. **Metric MDS**（经典 MDS：从距离矩阵恢复低维嵌入）
7. **Locally Linear Embedding (LLE)**（kNN + weight solve + eigendecomposition）
8. **Gaussian KDE**（高斯核密度估计，支持 `score_samples`/`pdf`）

## 代码结构

每个算法单独一个模块文件（避免 8 分支冲突）：

- `ml_algorithms/python/lasso.py`
- `ml_algorithms/python/elastic_net.py`
- `ml_algorithms/python/kernel_ridge.py`
- `ml_algorithms/python/gaussian_process.py`
- `ml_algorithms/python/kernel_pca.py`
- `ml_algorithms/python/mds.py`
- `ml_algorithms/python/lle.py`
- `ml_algorithms/python/kde.py`

风格约定：

- `dataclass` + `fit(...) -> self`（对齐现有实现）
- `np.float64` 作为主要 dtype
- 参数校验（维度、n_neighbors 等）尽量明确

## 测试与验收

新增一个专用测试文件：

- `tests/test_more_ml_algorithms_pack.py`

测试目标：

- 每个新模块可被发现并导入（用 `importlib.util.find_spec(...)` 做断言，避免 ImportError 直接 error）
- 核心接口输出 shape 正确、数值有限（`np.isfinite`）
- 用小规模合成数据验证“基本有效”（例如相关系数、MSE 阈值、密度单调性）

验收命令：

- `pytest -q tests/test_more_ml_algorithms_pack.py`
- `pytest -q`

## Git 工作流

- 整合分支：`feat/more-ml-algorithms-pack`
- 8 个功能分支（每个分支只新增 1 个模块文件）：
  - `feat/alg-lasso`
  - `feat/alg-elastic-net`
  - `feat/alg-kernel-ridge`
  - `feat/alg-gaussian-process`
  - `feat/alg-kernel-pca`
  - `feat/alg-mds`
  - `feat/alg-lle`
  - `feat/alg-kde`

合并策略：

- 算法实现分支全部 merge 回整合分支
- README 更新与测试文件只在整合分支改（避免冲突）
