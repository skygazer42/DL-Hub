---
icon: material/server-network
---

# 联邦学习 Zoo

> **36 策略 / 13 类别** --- 覆盖优化、个性化、隐私、压缩、安全聚合等联邦学习全链路策略。

---

## CLI 快速上手

```bash
# 列出全部联邦学习策略 ID
python -m zoo.fl --list

# 模糊搜索
python -m zoo.fl --search fed

# Smoke Test（策略验证）
python -m zoo.fl --smoke fedavg
```

---

## 全部 36 策略总览

| 类别 | 策略 | 说明 |
|:-----|:-----|:----|
| Optimization | FedAvg, FedProx, FedNova, FedDyn | 全局优化目标与本地训练的平衡 |
| Server Optimizer | FedAdam, FedYogi | 服务端自适应优化器 |
| Control Variate | SCAFFOLD | 方差缩减控制变量校正客户端漂移 |
| Feature Normalization | FedBN | 保留本地 BatchNorm 统计量 |
| Personalization | FedPer, APFL, Ditto, pFedMe, MOON, Per-FedAvg, FedRep, FedAMP, FedProto, IFCA | 个性化联邦学习 |
| Fairness | q-FedAvg, AFL, TERM | 公平性保障 |
| Long-tail Robustness | FedRS, FedLC, FedRoD | 长尾分布鲁棒性 |
| Split Learning | SplitFed, SplitFedV2 | 模型分割联邦学习 |
| Heterogeneous Width | HeteroFL, FjORD | 异构模型宽度适配 |
| Distillation | FedGKT, FedDF | 知识蒸馏联邦学习 |
| Privacy | DP-FedAvg, DP-FedProx | 差分隐私保护 |
| Compression | FedPAQ, STC | 通信压缩 |
| Secure Aggregation | SecureAgg, LightSecAgg | 安全聚合协议 |

---

## 详细分类

### Optimization

解决联邦学习中客户端数据异构 (Non-IID) 导致的优化挑战。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedAvg | 本地多步 SGD + 服务端加权平均，联邦学习基线 | `local_epochs`, `lr` |
| FedProx | 在本地损失中添加 Proximal Term 约束客户端漂移 | `mu` (正则系数) |
| FedNova | 归一化平均消除本地更新步数不一致的偏差 | `local_epochs` |
| FedDyn | 动态正则化对齐全局目标与本地目标 | `alpha` (动态正则系数) |

!!! tip "一行构建联邦策略"

    ```python
    from zoo.fl import build

    strategy = build("fedprox", mu=0.01)
    ```

---

### Server Optimizer

在服务端使用自适应优化器加速全局收敛。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedAdam | 服务端 Adam 优化器聚合伪梯度 | `server_lr`, `beta1`, `beta2` |
| FedYogi | 服务端 Yogi 优化器，对极端梯度更鲁棒 | `server_lr`, `beta1`, `beta2` |

---

### Control Variate

利用控制变量技术校正客户端漂移。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| SCAFFOLD | 客户端/服务端双控制变量，方差缩减 | `server_lr` |

!!! note "SCAFFOLD 核心思想"

    每个客户端维护一个控制变量 $c_i$，服务端维护全局控制变量 $c$。本地更新时加入校正项 $(c - c_i)$，显著减少客户端漂移。

---

### Feature Normalization

处理联邦学习中特征分布差异。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedBN | 各客户端保留本地 BatchNorm 参数，不参与聚合 | — |

---

### Personalization

个性化联邦学习 --- 在全局知识共享与本地个性化之间取得平衡。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedPer | 底层共享 + 顶层个性化 (Partial Model Sharing) | `n_shared_layers` |
| APFL | 自适应插值全局模型与本地模型 | `alpha` (插值系数) |
| Ditto | 多任务学习框架 + Proximal 正则化本地模型 | `lambda` (正则系数) |
| pFedMe | Moreau Envelope 正则化个性化联邦学习 | `lambda`, `K` |
| MOON | 对比学习对齐本地表示与全局表示 | `mu` (对比损失权重) |
| Per-FedAvg | MAML 风格元学习初始化 + 本地微调 | `meta_lr`, `inner_steps` |
| FedRep | 表示共享 + 分类头个性化，交替优化 | `head_epochs` |
| FedAMP | Attention-based Message Passing 个性化聚合 | `sigma` |
| FedProto | 原型网络风格，共享类别原型而非模型参数 | `lambda` |
| IFCA | 聚类联邦学习，多个全局模型服务不同客户端群 | `n_clusters` |

---

### Fairness

保障不同客户端性能公平性。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| q-FedAvg | 重加权聚合，对性能差的客户端赋予更大权重 | `q` (公平性参数) |
| AFL | Agnostic Federated Learning，最小化最坏客户端损失 | `lambda` |
| TERM | Tilted Empirical Risk Minimization，倾斜 ERM 平衡公平与效率 | `t` (倾斜参数) |

---

### Long-tail Robustness

处理联邦场景下的长尾类别分布。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedRS | 受限 Softmax 抑制客户端缺失类别的负面影响 | `alpha` |
| FedLC | 校准本地分类器消除类别先验偏差 | `tau` |
| FedRoD | 解耦通用头与个性化头，联合优化 | `hyper_l` |

---

### Split Learning

将模型在客户端与服务端之间分割执行。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| SplitFed | Split Learning + Federated Averaging 结合 | `split_layer` |
| SplitFedV2 | 改进 SplitFed，减少服务端计算与通信 | `split_layer` |

---

### Heterogeneous Width

支持不同客户端使用不同宽度的模型。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| HeteroFL | 自适应宽度子网络，按客户端算力分配模型大小 | `width_ratios` |
| FjORD | Ordered Dropout 有序丢弃，渐进式子网络提取 | `p_list` |

---

### Distillation

利用知识蒸馏实现异构模型联邦学习。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedGKT | Group Knowledge Transfer --- 边缘小模型蒸馏至服务端大模型 | `T` (温度) |
| FedDF | 服务端 Ensemble Distillation，聚合客户端 Logits | `T`, `public_data_size` |

---

### Privacy

基于差分隐私 (Differential Privacy) 保护客户端数据。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| DP-FedAvg | FedAvg + 高斯噪声裁剪 (Gaussian Mechanism) | `epsilon`, `delta`, `clip_norm` |
| DP-FedProx | FedProx + 差分隐私保护 | `epsilon`, `delta`, `clip_norm`, `mu` |

!!! warning "隐私预算提示"

    差分隐私中 $\epsilon$ 越小隐私保护越强，但模型精度下降越明显。实践中通常取 $\epsilon \in [1, 10]$。

---

### Compression

减少联邦学习中的通信开销。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| FedPAQ | 周期性平均 + 量化压缩 | `n_bits` (量化位数) |
| STC | Sparse Ternary Compression --- 稀疏三值压缩 | `top_k_ratio` |

---

### Secure Aggregation

密码学安全聚合，服务端无法窥探单个客户端更新。

| 策略 | 核心创新 | 关键参数 |
|:-----|:---------|:---------|
| SecureAgg | Secret Sharing + Masking 安全聚合协议 | `threshold` |
| LightSecAgg | 轻量级安全聚合，降低通信与计算开销 | `threshold` |

---

## 用法示例

=== "基础联邦训练"

    ```python
    from zoo.fl import build

    strategy = build("fedavg")
    # strategy.aggregate(client_updates) -> global_update
    ```

=== "个性化联邦"

    ```python
    from zoo.fl import build

    strategy = build("ditto", lambda_reg=0.1)
    # strategy.local_train(model, data) -> personalized_model
    ```

=== "差分隐私"

    ```python
    from zoo.fl import build

    strategy = build("dp_fedavg", epsilon=8.0, delta=1e-5, clip_norm=1.0)
    # 自动裁剪 + 加噪
    ```

=== "CLI 验证"

    ```bash
    python -m zoo.fl --smoke scaffold
    ```
