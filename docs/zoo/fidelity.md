---
icon: material/scale-balance
---

# Model Zoo 保真度审计

DL-Hub 的注册数量回答“统一接口能构建多少个 ID”，保真度审计回答“源码是否体现注册名所指方法的关键机制”。两者是不同指标：**注册 ID 不是独立论文复现数量**。

当前台账只记录已读源码并形成证据的审计组。没有出现在台账里的普通实现统一视为
`unreviewed`；但源码直接执行 `return build_baseline_*` 已经足以保守判定为
source-inferred `baseline-alias`，不需要假装其状态未知。

此外，仓库维护一份由 AST 直接从当前源码生成的
[`baseline-inventory.json`](baseline-inventory.json)：凡是执行
`return build_baseline_*` 的入口都会出现在清单中。显式审计台账给出已读源码的结论，
baseline inventory 则区分逐组 `reviewed` 和仅凭直接委托事实得到的 `source-inferred`，保证
所有共享委托都不会隐身，也不会把源码推断冒充人工机制审计。

这里的实现等级与数据来源、验证范围相互独立；三者的统一定义和课程到审计的完整链路见
[实现契约：从课程到可验证系统](../implementation-contract.md)。

## 分级标准

| 等级 | 含义 | 不代表什么 |
|:-----|:-----|:-----------|
| `reference` | 已核对注册名的关键计算机制，源码与稳定接口中存在可观察的对应实现 | 不保证复现论文全部训练配方、数据、权重与指标 |
| `compact` | 保留关键计算机制，同时缩小参数量、输入规模或训练配方，便于本地运行与研究迭代 | 不能直接与论文完整配置的 benchmark 结果等同 |
| `baseline-alias` | 论文/方法名作为兼容入口，但当前计算明确委托给通用基线 | 不是该论文的机制复现，也不参与复现数量统计 |
| `unreviewed` | 尚未完成源码级核对 | 不应被推断为任何其他等级 |

## 当前结果

当前台账优先覆盖高审计压力且大量复用共享实现的区域，共 **22 个审计组、232 个源码入口**：

| 审计组 | 等级 | 入口数 | 当前源码结论 |
|:-------|:-----|------:|:-------------|
| Detection：5 个 DETR 论文标签 | `baseline-alias` | 5 | 共享通用紧凑 DETR 基线；窗口注意力、prompt、开放词表文本分类等标志机制缺失 |
| Temporal Action Localization | `baseline-alias` | 10 | 全部委托给堆叠 GRU 公共基线，`depth` 已生效，但 `family` 仍不改变计算机制 |
| Video Temporal Grounding | `baseline-alias` | 10 | 全部委托给堆叠 GRU 边界头，缺少 query decoder / 显式跨模态注意力 |
| Referring Expression Comprehension | `baseline-alias` | 10 | 共用卷积编码器、文本线性投影与加法融合，没有各家视觉语言融合机制 |
| Open-Vocabulary Segmentation | `baseline-alias` | 10 | 共用 text-bias 紧凑基线，没有预训练视觉语言空间与类别文本相似度分类 |
| Referring Expression Segmentation | `baseline-alias` | 10 | 与上项共用同一 text-bias 基线，缺少 token 级语言定位与家族差异 |
| QueryFocus-Sum | `compact` | 1 | 已支持 query 向量、query-frame 对齐与条件化打分；尚无 token 级 query encoder / cross-attention |
| MemoryTokenSum | `compact` | 1 | 已加入可学习 memory token、frame-to-memory 更新与 memory-to-frame 读取；尚无跨视频持久状态 |
| SegmentFormer-Sum | `compact` | 1 | 已有多窗口 segment pooling，但没有 Transformer 交互 |
| CLIP-CoSeg | `compact` | 1 | 已支持文本特征与归一化图文相似度调制；尚无预训练 CLIP 编码器、tokenizer 与对比预训练 |
| Token-Affinity CoSeg | `compact` | 1 | 有图像级描述子注意力，但没有空间 token-to-token affinity |
| Blur Detection | `baseline-alias` | 10 | 全部委托给 `TinyBlurDetector`；多个论文/算子标签走相同分支，其余只切换局部模式 |
| Crack Detection | `baseline-alias` | 10 | 全部委托给 `TinyCrackDetector`；缺少 U-Net、FPN、HED、轮廓与骨架等命名机制 |
| Image Retrieval | `compact` | 10 | 已按标签选择 angular、context、contrastive、local-global、GeM、VLAD、pairwise、proxy、regional 或 Transformer 路径 |
| Visual Place Recognition | `compact` | 10 | 已按标签选择 adaptive GeM、proxy、local-global、geo context、选择性扫描、MixVPR、pairwise、patch VLAD、regional 或 Transformer 路径 |
| Fine-Grained Retrieval | `compact` | 10 | 已按标签选择 bilinear、global、context part、multiscale、选择性扫描、part VLAD、prompt、regional、part token 或 Transformer 路径 |
| Layout Generation | `compact` | 10 | 十个入口已分别实现潜变量残差生成、VAE 空间瓶颈、自注意力、坐标 objectness、多尺度融合、轴向混合、约束投影、关系注意力、时间条件去噪和输入依赖选择性扫描；仍不是完整论文训练配方 |
| Point-Cloud Registration | `compact` | 10 | 十个入口已分别实现全局迭代特征、交叉对应、联合 Transformer、Sinkhorn、软混合模型、圆柱描述子、粗到细、几何注意力、重叠加权和径向顺序选择性扫描；仍缺完整刚体求解与真实 benchmark |
| VLM Representative Paths | `compact` | 12 | 代表家族已接收真实图像/Token，并分别执行对比双编码器、联合多模态 Transformer、跨注意力融合或 query-token bridge；生成 logits 逐位置产生 |
| VLM Shared Labels | `baseline-alias` | 58 | 已具备真实输入与四类可执行多模态路径，但产品/论文标签仍主要只映射到共享 mode 与 flag，缺少各命名模型机制 |
| Diffusion Representative Paths | `compact` | 10 | 代表家族已接收显式 `x_t` / timestep，分别执行空间卷积、patch Transformer 或 latent autoencoder 去噪，并使用 mode-specific 多步更新 |
| Diffusion Shared Labels | `baseline-alias` | 22 | 已有真实状态、时间/标签条件和迭代采样，但标签仍主要映射到三类共享 denoiser 与五种 prediction mode |

`reference` 当前为 0 个已审计组。这不表示仓库中不存在参考级实现，只表示当前台账没有在
缺少完整核对时提前授予该等级。

### 全量 baseline wrapper 清单

当前源码共有 **1,970** 个直接 baseline wrapper；Layout Generation、Point-Cloud Registration、
三个 Retrieval 包、12 个代表 VLM 路径与 10 个代表 Diffusion 路径已经从首次快照中移除
72 个直接委托：

| 领域 | wrapper | 当前显式审计 |
|:--|--:|--:|
| Vision | 1,660 | 20 个已审计 wrapper 分布在现有审计组中 |
| Multimodal | 148 | 58 个共享 mode 的 VLM 标签已显式审计为 `baseline-alias` |
| Generative | 92 | 22 个共享 mode 的 Diffusion 标签已显式审计为 `baseline-alias` |
| Point Cloud | 70 | 0 |
| **合计** | **1,970** | **100 reviewed / 1,870 source-inferred aliases / 0 unidentified** |

这里的 100 是“232 个已审计源码入口”与 direct baseline wrapper 清单的交集。其余 1,870 个
条目虽然尚未逐组核对命名机制，但直接 baseline 委托这一源码事实已足以确定它们不能被当作
独立论文实现；因此 level 为 `baseline-alias`、review status 为 `source-inferred`。其余已审计入口
可能是已有独立机制的 compact 源码，或不是这种直接委托形态。清单逐项记录源码路径、helper、
行号、当前 fidelity level 和审计 key，不能用统计汇总代替逐文件查询。

`make fidelity` 会重新解析源码并与受版本控制的 JSON 比较。新增、删除、移动 wrapper，修改 helper、
行号或显式等级，都会在清单刷新前使 CI 失败；wrapper 总数也被锁定为只降不升的债务基线。

### Retrieval 机制升级的共享源码链路

三个 retrieval 包的 30 个入口仍复用统一的构建基础设施，但 `family` 现在会选择真实计算路径：

```text
方法标签 builder
  -> 对应包的 _common.py 重新导出
  -> dlhub/vision/_shared/retrieval.py::build_compact_retrieval_model
  -> CompactRetrievalModel
  -> family 对应的 pooling / conditioning / scoring 模块
```

当前路径包含 learnable GeM、soft-residual VLAD、局部/全局显著性、区域和多尺度聚合、
bilinear interaction、part token、空间 Transformer、输入依赖选择性扫描、外部 context
条件化、pairwise scorer、proxy refinement、angular 与 temperature scoring。行为测试验证
context 会改变 embedding、VLAD/attention 权重归一化、pairwise scorer 不等同于点积，并覆盖
反向传播。`compact` 仍只表示关键类别机制已存在；没有预训练文本编码器、论文训练目标、权重和
真实 benchmark 证据，因此不授予 `reference`。

### VLM 真实输入与分级边界

VLM 公共核心不再把内部随机图像当成唯一输入：`forward` 现在接受调用者提供的 `images`、
`input_ids` 和 `instruction_ids`，并保留无输入时的兼容数据生成。四种 mode 分别执行对比双编码、
联合序列 Transformer、text-to-image cross-attention 和两阶段 query-token bridge；启用生成时会
输出 `[batch, sequence, vocabulary]` 的逐位置 `token_logits`，不再复制单个分类向量。

行为检查用固定真实输入验证确定性、图像/文本敏感性、注意力归一化、instruction 条件作用、
逐位置生成和反向传播，并对 70 个 family 的 tiny 注册逐一执行前向。12 个与这些核心机制直接对应的
代表入口列为 `compact`；其余 58 个仍是 `baseline-alias`，因为 OCR、视频、grounding、视觉专家、
高分辨率或具体 LLM adapter 等标签机制并未由四种共享 mode 自动获得。

### Diffusion 显式去噪状态与采样边界

Diffusion 公共核心现在接受 `x_t`、`timesteps` 和可选 `labels`；给定相同状态与时间时前向确定，
时间、标签和 noisy state 的变化都会传入去噪网络。像素路径使用 time-conditioned residual conv，
DiT 类路径使用带二维位置的 patch Transformer，latent 路径使用卷积编码、latent FiLM block 与
解码器。`sample()` 对显式或随机 initial noise 运行可配置的多步 schedule，`step_scale` 不再是
DDIM 等入口的死参数。

行为检查覆盖三类架构、时间和标签敏感性、一步/多步采样差异、DDPM/DDIM update 差异、反向传播，
并用显式 noisy input 逐一运行 32 个 family 的 tiny 注册。10 个代表入口列为 `compact`；其余
22 个仍为 `baseline-alias`，因为精确 SDE/ODE、solver、文本编码、mask/audio、高分辨率骨干、
训练目标和权重尚未按命名方法实现。

## 可复核接口

台账源位于 `dlhub/zoo_fidelity.py`，包含稳定 key、等级、源码路径、缺失机制、证据和下一步。普通查询不导入模型；`--check` 只导入注册表统计 ID，不实例化或运行模型：

```bash
# 校验 key、等级、证据和源码路径
python scripts/model_fidelity.py --check

# 查看全部或筛选高风险组
python scripts/model_fidelity.py --list
python scripts/model_fidelity.py --list --level baseline-alias

# 查看单组证据，或供其他工具读取
python scripts/model_fidelity.py --show vision.detection.detr-paper-labels
python scripts/model_fidelity.py --show vision.image-retrieval.mechanism-aware-compact
python scripts/model_fidelity.py --json

# 机制升级或 wrapper 变化后，显式刷新并审核全量清单差异
python scripts/model_fidelity.py --write-baseline-inventory
python scripts/model_fidelity.py --check
```

CI 通过 `make fidelity` 执行同一校验。新增注册组不会自动获得已审计等级；直接委托 baseline 的
源码会自动成为 source-inferred `baseline-alias` 清单项；只有完成逐组源码核对后才加入显式
审计台账并获得 audit key。

校验同时计算“审计压力”：`全部注册 ID / 已审计源码入口`。首轮债务基线是
**8611 / 80 = 107.64**；Retrieval 审计先把分母提高到 110，本轮 Layout Generation
机制升级再把分母提高到 120；Point-Cloud Registration 提高到 130，VLM 全量源码分级提高到
200，Diffusion 全量源码分级继续提高到 232，并把上限同步收紧为
**8611 / 232 = 37.12**。
`--check` 会使用仓库当时的实际注册数重新计算，因此后续注册增长
仍必须伴随足够的源码审计，不能重新消耗已经偿还的债务。这个上限是只降不升的债务棘轮，
不是质量目标；修复方式只能是补充有证据的源码审计，或减少没有足够支撑的注册项。

## 后续优化顺序

1. 共享实现使用明确的领域 baseline 命名；论文名在标志机制落地前保留 `baseline-alias` 标识。
2. 按影响面逐族补机制：先 query/text conditioning，再 attention / state-space / proposal 结构。
3. 每完成一族，用最小行为检查确认机制生效后再升级等级；不以改类名或复制文件作为“完成”。
