# 无监督学习算法

本页列出 DL-Hub 中所有纯 NumPy 实现的无监督学习算法。
每个算法均位于 `ml_algorithms/python/` 目录下，采用 `@dataclass` + `fit/transform/predict` 接口。

---

## 聚类

### K-Means

> 文件：`kmeans.py`

经典的**质心迭代**聚类算法（Lloyd 算法）。

- 初始化：随机选取 K 个质心
- E 步：将每个样本分配到最近质心
- M 步：重新计算各簇质心
- 迭代直到收敛

```python
from ml_algorithms.python.kmeans import KMeans

model = KMeans(n_clusters=3, n_iters=100)
model.fit(X)
labels = model.predict(X)
```

### K-Medoids

> 文件：`kmedoids.py`

使用真实数据点作为 **Medoid** 中心的聚类算法（PAM 算法）。

- 与 K-Means 不同，中心点必须是数据集中的实际样本
- 对离群值更鲁棒
- 通过交换 Medoid 与非 Medoid 点来优化目标函数

### Agglomerative Clustering

> 文件：`clustering.py`

自底向上的**层次聚类**算法。

- 从每个样本作为独立簇开始
- 逐步合并距离最近的两个簇
- Linkage 策略：single / complete / average
- 最终形成树状聚类结构（dendrogram）

### DBSCAN

> 文件：`clustering.py`

基于**密度**的聚类算法，能发现任意形状的簇。

- 核心点：邻域内样本数 ≥ `min_samples`
- 边界点：在核心点邻域内但自身不是核心点
- 噪声点：既不是核心点也不是边界点
- 通过邻域扩展连接密度可达的点

### Spectral Clustering

> 文件：`spectral_clustering.py`

基于**图拉普拉斯**特征向量的聚类方法。

- 构建相似度图（高斯核）
- 计算归一化 Laplacian 矩阵
- 取前 K 个最小特征值对应的特征向量
- 在特征向量空间中应用 K-Means

---

## 降维

### PCA

> 文件：`pca.py`

主成分分析，通过**特征值分解**实现降维。

- 目标：找到方差最大化的投影方向
- 计算协方差矩阵的特征值和特征向量
- 选取前 K 个主成分作为新的基
- 广泛用于数据可视化和预处理

```python
from ml_algorithms.python.pca import PCA

model = PCA(n_components=2)
model.fit(X)
X_reduced = model.transform(X)
```

### NMF

> 文件：`nmf.py`

**非负矩阵分解**，将矩阵分解为两个非负矩阵的乘积。

- $V \approx W \cdot H$，其中 $W, H \geq 0$
- 使用乘法更新规则保证非负性
- 常用于主题建模、图像分解

### FastICA

> 文件：`ica.py`

**独立成分分析**，从混合信号中分离出独立源信号。

- 基于信号统计独立性假设
- 使用 Fixed-point 迭代算法
- 最大化非高斯性（negentropy）
- 经典应用：鸡尾酒会问题

### Isomap

> 文件：`isomap.py`

基于**测地距离**的非线性降维方法。

- 构建 K 近邻图
- 使用 Dijkstra 计算测地距离
- 对测地距离矩阵应用 MDS（多维缩放）
- 保留数据的全局几何结构

---

## 概率模型

### GMM

> 文件：`gmm.py`

**高斯混合模型**，使用 EM 算法拟合多个高斯分布的混合。

- E 步：计算每个样本属于各高斯分量的后验概率
- M 步：更新均值、协方差和混合权重
- 可用于聚类和密度估计
- 比 K-Means 更灵活（软分配、椭圆形簇）

---

## 序列模型

### Markov Chain

> 文件：`markov_chain.py`

离散**马尔可夫链**，建模状态之间的转移概率。

- 转移矩阵：$P(s_{t+1} | s_t)$
- 平滑处理避免零概率
- 可生成序列样本

### N-gram Language Model

> 文件：`ngram.py`

基于**计数**的统计语言模型。

- 通过 N-gram 频率估计条件概率
- Laplace 平滑处理未见过的 N-gram
- 支持文本生成和困惑度评估

### Categorical HMM

> 文件：`hmm.py`

离散观测的**隐马尔可夫模型**。

- Forward 算法：计算观测序列概率
- Viterbi 算法：解码最优隐状态序列
- 全部在 **log-space** 下计算，避免数值下溢
- 参数：初始概率、转移概率、发射概率

!!! tip "数值稳定性"

    HMM 实现中所有概率运算都在 log-space 下进行，
    使用 log-sum-exp 技巧确保长序列的数值稳定性。
