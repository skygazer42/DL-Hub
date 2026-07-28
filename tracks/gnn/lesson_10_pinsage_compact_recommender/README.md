# Lesson 10：PinSAGE 风格（compact 推荐图，item embedding）

这节课用一个很小的「用户-物品」二部图跑通 PinSAGE 的核心直觉：

- 通过 **item → user → item** 的随机游走，把每个 item 的邻居（相似 item）抽样出来
- 用 **GraphSAGE 风格的聚合**把邻居信息融入 item 表示
- 用 **negative sampling** 做无监督训练（让相似 item 更接近）
- 用一个很轻量的 top-k 指标做 sanity check（是否能把同一用户偏好的 item 拉近）

## 运行方式

快速冒烟（CPU，几秒内）：

```bash
python -m tracks.gnn.lesson_10_pinsage_compact_recommender.train \
  --device cpu --epochs 3 --run-name smoke
```

## 输出产物（统一规范）

`outputs/gnn/lesson_10_pinsage_compact_recommender/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `embeddings.pt`：`(num_items, D)` 的 item 表示 + 邻居表
- `logs/train.log`
- `checkpoints/checkpoint.pt`

