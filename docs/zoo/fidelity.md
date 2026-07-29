---
icon: material/scale-balance
---

# Model Zoo 保真度审计

DL-Hub 的注册数量回答“统一接口能构建多少个 ID”，保真度审计回答“源码是否体现注册名所指方法的关键机制”。两者是不同指标：**注册 ID 不是独立论文复现数量**。

当前台账只记录已读源码并形成证据的审计组。没有出现在台账里的实现统一视为 `unreviewed`，既不默认判为忠实，也不默认判为有问题。

这里的实现等级与数据来源、验证范围相互独立；三者的统一定义和课程到审计的完整链路见
[实现契约：从课程到可验证系统](../implementation-contract.md)。

## 分级标准

| 等级 | 含义 | 不代表什么 |
|:-----|:-----|:-----------|
| `reference` | 已核对注册名的关键计算机制，源码与稳定接口中存在可观察的对应实现 | 不保证复现论文全部训练配方、数据、权重与指标 |
| `compact` | 保留关键计算机制，同时缩小参数量、输入规模或训练配方，便于本地运行与研究迭代 | 不能直接与论文完整配置的 benchmark 结果等同 |
| `baseline-alias` | 论文/方法名作为兼容入口，但当前计算明确委托给通用基线 | 不是该论文的机制复现，也不参与复现数量统计 |
| `unreviewed` | 尚未完成源码级核对 | 不应被推断为任何其他等级 |

## 首批结果

本轮优先检查先前数值审核暴露出的高风险区域，共 **13 个审计组、80 个源码入口**：

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

`reference` 当前为 0 个已审计组。这不表示仓库中不存在参考级实现，只表示本轮没有在缺少完整核对时提前授予该等级。

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
python scripts/model_fidelity.py --json
```

CI 通过 `make fidelity` 执行同一校验。新增注册族时不会自动获得任何等级；完成源码核对后，才把记录加入台账。

校验同时计算“审计压力”：`全部注册 ID / 已审计源码入口`。上限锁定在本轮债务基线 **8611 / 80 = 107.64**，因此即使只增加一个注册 ID，只要没有同步增加已审计源码入口，CI 就会失败。这个上限是只降不升的债务棘轮，不是质量目标；修复方式只能是补充有证据的源码审计，或减少没有足够支撑的注册项。

## 后续优化顺序

1. 共享实现使用明确的领域 baseline 命名；论文名在标志机制落地前保留 `baseline-alias` 标识。
2. 按影响面逐族补机制：先 query/text conditioning，再 attention / state-space / proposal 结构。
3. 每完成一族，用最小行为检查确认机制生效后再升级等级；不以改类名或复制文件作为“完成”。
