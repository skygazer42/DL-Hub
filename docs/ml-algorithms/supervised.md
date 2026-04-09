# 监督学习算法

本页列出 DL-Hub 中所有纯 NumPy 实现的监督学习算法。
每个算法均位于 `ml_algorithms/python/` 目录下，采用 `@dataclass` + `fit/predict` 接口。

---

## 线性模型

### Linear Regression

> 文件：`linear_models.py`

经典线性回归，支持**最小二乘闭式解**和**梯度下降**两种求解方式。

- 损失函数：均方误差 (MSE)
- 闭式解：$\mathbf{w} = (X^T X)^{-1} X^T y$
- 梯度下降：逐步更新权重直到收敛

### Ridge Regression

> 文件：`linear_models.py`

在 Linear Regression 基础上添加 **L2 正则化**，防止过拟合。

- 闭式解：$\mathbf{w} = (X^T X + \lambda I)^{-1} X^T y$
- 正则化强度由超参数 $\lambda$ 控制

### Logistic Regression

> 文件：`linear_models.py`

二分类线性模型，使用 **Sigmoid** 函数将线性输出映射到概率。

- 损失函数：二元交叉熵 (Binary Cross-Entropy)
- 决策边界：线性超平面
- 优化：梯度下降

### Softmax Regression

> 文件：`linear_models.py`

Logistic Regression 的多分类扩展，使用 **Softmax** 函数。

- 损失函数：多分类交叉熵 (Categorical Cross-Entropy)
- 输出：各类别概率分布
- 归一化：$\text{softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$

---

## 核方法

### Linear SVM

> 文件：`svm.py`

支持向量机，通过 **Hinge Loss** 实现最大间隔分类。

- 目标：最大化决策边界到最近样本的间隔
- 损失函数：$\max(0, 1 - y \cdot f(x))$
- 优化：次梯度下降

---

## 树模型

### Decision Tree

> 文件：`decision_tree.py`

基于 **Gini 不纯度** 进行递归分裂的决策树。

- 分裂准则：选择使 Gini 不纯度下降最大的特征和阈值
- 递归构建：直到满足停止条件（最大深度、最小样本数等）
- 支持分类和回归

---

## 集成方法

### Random Forest

> 文件：`random_forest.py`

基于 **Bagging** 的集成方法，每棵树使用数据和特征的随机子集。

- Bootstrap 采样：有放回地抽取训练样本
- 特征随机采样：每次分裂只考虑部分特征
- 预测：多棵树投票 / 平均

### AdaBoost

> 文件：`adaboost.py`

自适应 **Boosting** 算法，使用加权 Decision Stumps 作为弱学习器。

- 每轮迭代增加被错分样本的权重
- 弱学习器加权组合为强学习器
- 学习器权重与其准确率成正比

### Gradient Boosting

> 文件：`gradient_boosting.py`

梯度提升算法，每棵新树拟合前一轮的**残差**。

- 沿损失函数的负梯度方向逐步优化
- 支持自定义损失函数
- 学习率控制每步更新幅度

---

## 概率模型

### Naive Bayes

> 文件：`naive_bayes.py`

基于贝叶斯定理的分类器，假设特征之间**条件独立**。

- 后验概率：$P(y|x) \propto P(y) \prod_i P(x_i|y)$
- Laplace 平滑避免零概率
- 高维稀疏数据表现良好

---

## 生成模型

### LDA / QDA

> 文件：`discriminant_analysis.py`

线性判别分析 (LDA) 和二次判别分析 (QDA)，基于**高斯假设**的生成式分类器。

- LDA：假设各类别共享协方差矩阵 → 线性判别函数
- QDA：各类别拥有独立协方差矩阵 → 二次判别函数
- 分类决策：比较各类别后验概率

---

## 近邻方法

### KNN

> 文件：`knn.py`

K 最近邻算法，基于**距离度量**进行预测。

- 分类：K 个近邻的多数投票
- 回归：K 个近邻的均值
- 距离度量：欧氏距离

---

## 神经网络

### Perceptron

> 文件：`perceptron.py`

单层感知机，使用**感知机学习规则**进行训练。

- 线性可分数据收敛保证
- 更新规则：误分类时调整权重
- 激活函数：阶跃函数

### MLP

> 文件：`mlp.py`

多层感知机，使用**反向传播**算法训练。

- 前向传播：逐层计算激活值
- 反向传播：基于链式法则计算梯度
- 激活函数：ReLU / Sigmoid
- 支持任意隐藏层配置
