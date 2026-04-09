# 优化工具箱

纯 NumPy 实现 — 理解优化器和调度器的数学本质。

DL-Hub 的优化工具箱分为两部分：

1. **深度学习优化组件** (`optimization/python/`) — Optimizer、LR Scheduler、Loss、Metric
2. **元启发式优化算法** (`optimization/`) — ACO、GA、PSO、AHP、Lasso

---

## 深度学习优化组件

### Optimizers

所有优化器实现位于 `optimization/python/optimizers.py`。

| 优化器 | 核心思想 | 特点 |
|--------|----------|------|
| **SGD** | 随机梯度下降 | 最基础的优化器 |
| **Momentum** | 动量加速 | 累积历史梯度方向，加速收敛 |
| **RMSProp** | 自适应学习率 | 对梯度平方的指数移动平均 |
| **Adagrad** | 自适应学习率 | 累积梯度平方和，自动衰减 |
| **Adam** | Momentum + RMSProp | 一阶/二阶矩估计 + 偏差修正 |

```python
from optimization.python.optimizers import Adam

optimizer = Adam(params, lr=1e-3, betas=(0.9, 0.999))
optimizer.step(grads)
```

### LR Schedulers

所有调度器实现位于 `optimization/python/lr_schedulers.py`。

| 调度器 | 策略 | 公式 |
|--------|------|------|
| **StepDecay** | 阶梯衰减 | 每 N 步乘以 $\gamma$ |
| **ExponentialDecay** | 指数衰减 | $\text{lr} = \text{lr}_0 \cdot \gamma^t$ |
| **CosineAnnealing** | 余弦退火 | $\text{lr} = \frac{1}{2}(1 + \cos(\pi t / T)) \cdot \text{lr}_0$ |
| **WarmupCosine** | 预热 + 余弦 | 线性 Warmup 后接 Cosine Annealing |

### Losses

所有损失函数实现位于 `optimization/python/losses.py`。

| 损失函数 | 适用场景 |
|----------|----------|
| **MSE** (均方误差) | 回归任务 |
| **MAE** (平均绝对误差) | 回归任务（对离群值更鲁棒） |
| **Binary Cross-Entropy** | 二分类 |
| **Categorical Cross-Entropy** | 多分类 |

### Metrics

所有评估指标实现位于 `optimization/python/metrics.py`。

| 指标 | 说明 |
|------|------|
| **Accuracy** | 分类准确率 |
| **Precision** | 精确率 |
| **Recall / F1** | 召回率 / F1 分数 |
| **R² Score** | 回归决定系数 |

---

## 元启发式优化算法

!!! note "历史实现"

    以下元启发式算法为项目早期实现，部分使用 MATLAB 编写，保留作为参考。

### 蚁群优化 ACO

> 目录：`optimization/ACO/`

蚁群优化算法，用于求解**旅行商问题** (TSP)。

- 模拟蚂蚁在图上行走寻找最短路径
- 信息素更新：好路径上沉积更多信息素
- 启发式信息：距离的倒数

### 遗传算法 GA

> 目录：`optimization/GA/`

基于生物进化思想的**进化搜索**算法。

- 编码：将解表示为染色体
- 选择：适应度函数选择优秀个体
- 交叉：组合父代基因产生后代
- 变异：随机扰动维持种群多样性

### 粒子群优化 PSO

> 目录：`optimization/PSO/`

模拟鸟群/鱼群行为的**群体智能优化**算法。

- 每个粒子维护自身历史最优位置
- 全局最优引导粒子运动方向
- 惯性权重平衡探索与开发
- 支持 1D 和 2D 优化

### 层次分析法 AHP

> 目录：`optimization/AHP/`

**多准则决策**方法，通过构建层次结构进行定量分析。

- 构建判断矩阵
- 一致性检验
- 计算权重向量

### Lasso 优化

> 目录：`optimization/Lasso/`

**L1 正则化路径**求解，用于特征选择和稀疏模型。

- 坐标下降法
- 正则化路径追踪
- 自动特征选择
