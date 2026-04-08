# ML 算法总览

27 个纯 NumPy 手写经典机器学习算法 — 零深度学习依赖，理解算法本质。

所有实现位于 `ml_algorithms/python/`，统一采用 `@dataclass` 模式，
接口风格与 scikit-learn 保持一致（`fit` / `predict` / `transform`），
但内部全部使用 NumPy 从零实现。

---

## 算法总表

| 类别 | 算法 | 文件 | 核心原理 |
|------|------|------|----------|
| **线性模型** | Linear Regression | `linear_models.py` | 最小二乘 / 梯度下降 |
| | Ridge Regression | `linear_models.py` | L2 正则化，闭式解 |
| | Logistic Regression | `linear_models.py` | Sigmoid，交叉熵损失 |
| | Softmax Regression | `linear_models.py` | Softmax，多分类交叉熵 |
| | Lasso Regression | `lasso.py` | L1 正则化，坐标下降 |
| | Elastic Net | `elastic_net.py` | L1 + L2 混合正则化 |
| | Kernel Ridge | `kernel_ridge.py` | 核技巧 + Ridge |
| **核方法** | Linear SVM | `svm.py` | Hinge Loss，最大间隔 |
| **树模型** | Decision Tree | `decision_tree.py` | Gini 不纯度，递归分裂 |
| **集成方法** | Random Forest | `random_forest.py` | Bagging，特征随机采样 |
| | AdaBoost | `adaboost.py` | Boosting，Decision Stumps |
| | Gradient Boosting | `gradient_boosting.py` | Boosting，残差拟合 |
| **概率模型** | Naive Bayes | `naive_bayes.py` | 条件独立假设，平滑 |
| | GMM | `gmm.py` | EM 算法，高斯混合 |
| **生成模型** | LDA | `discriminant_analysis.py` | 高斯假设，线性判别函数 |
| | QDA | `discriminant_analysis.py` | 高斯假设，二次判别函数 |
| **近邻** | KNN | `knn.py` | 距离度量，多数投票 |
| **聚类** | K-Means | `kmeans.py` | 质心迭代，Lloyd 算法 |
| | K-Medoids | `kmedoids.py` | Medoid 中心，PAM |
| | Agglomerative | `clustering.py` | 层次聚类，Linkage |
| | DBSCAN | `clustering.py` | 密度聚类，邻域扩展 |
| | Spectral Clustering | `spectral_clustering.py` | 图拉普拉斯，特征向量 |
| **降维** | PCA | `pca.py` | 特征值分解，方差最大化 |
| | NMF | `nmf.py` | 非负分解，乘法更新 |
| | FastICA | `ica.py` | 独立成分，Fixed-point |
| | Isomap | `isomap.py` | 测地距离，MDS |
| **序列模型** | Markov Chain | `markov_chain.py` | 转移矩阵，平滑 |
| | N-gram LM | `ngram.py` | 计数，Laplace 平滑 |
| | Categorical HMM | `hmm.py` | Forward / Viterbi，log-space |
| **神经网络** | Perceptron | `perceptron.py` | 感知机学习规则 |
| | MLP | `mlp.py` | 反向传播，链式法则 |

!!! info "额外算法"

    以下算法作为扩展包收录，不计入 27 个核心算法：

    - **Gaussian Process** (`gaussian_process.py`) — 高斯过程回归
    - **Kernel PCA** (`kernel_pca.py`) — 核主成分分析
    - **MDS** (`mds.py`) — 多维缩放
    - **LLE** (`lle.py`) — 局部线性嵌入
    - **KDE** (`kde.py`) — 核密度估计

---

## 快速体验

```python
from ml_algorithms.python.linear_models import LinearRegression
import numpy as np

X = np.random.randn(100, 3)
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * np.random.randn(100)

model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)
preds = model.predict(X)
```

---

## 详细文档

- [监督学习算法](supervised.md) — 14 个有标签学习算法详解
- [无监督学习算法](unsupervised.md) — 13 个无标签学习算法详解
