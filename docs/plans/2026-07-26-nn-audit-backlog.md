# 2026-07-26 神经网络审核遗留事项（待决策）

四路数值审核（共享构件 / 首次可运行包 / 经典 lesson / Llms+consistency）已完成，
9 处 CONFIRMED_BUG 已修复。以下为已确认但**有意不动**的事项，属产品决策而非 bug：

## 1. Zoo "一个玩具实现挂多个论文名"（范围最大）

同一份玩具实现按不同论文名注册多个变体，`family` 只是字符串标签，不改变计算：

- detection：focal/swin/align/prompt/open_vocab_detr 五个文件除类名外逐字节相同；
  swin_detr 无窗口注意力，open_vocab_detr 是闭集 Linear 分类头，prompt_detr 无 prompt。
- temporal_action_localization：actionformer/temporalmaxer/tridet/mambatal 等全是
  同一个单层 GRU + 双线性头（actionformer 无注意力、mambatal 无 SSM）。
- video_temporal_grounding：momentdetr/qdetr_ground 无 query/decoder 机制。
- referring_expression_comprehension：transvg/reftr 无 transformer 融合。
- video_summarization：queryfocus_sum 已于 2026-07-28 补上 query 条件化并转为
  compact-inspired；memorytokensum 已补上双向 memory token 读写并转为 compact-inspired；
  segmentformer 已有 segment pooling 但无 Transformer。
- open_vocabulary_segmentation / referring_expression_segmentation：20 个名字对应
  完全相同的模型。
- co_segmentation：clip_coseg 已于 2026-07-28 补上文本输入和图文相似度调制，
  转为 compact-inspired，但仍无预训练 CLIP 编码器；token_affinity 只有图像级亲和。

可选处理：a) 保持现状（zoo 定位是统一玩具脚手架，README/docs 说明即可）；
b) 给差异化程度最低的家族补上名义机制的最小实现；c) 在各包 docstring 声明
"变体仅作注册名区分"。

## 2. TAL / VTG 的 `depth` 参数无效（2026-07-28 已解决）

两个 `_common.py` 现已将 `depth` 传给 GRU 的 `num_layers`；tiny/small/base
分别使用 1/2/3 层，参数量和实际时序深度都会随配置变化。旧版 small/base
checkpoint 因新增 GRU 层参数，不能再直接 strict load。

## 3. lesson_09 compact RLHF PPO 的冻结 old-policy

ratio 的 old policy 是永不更新的初始参考模型、样本来自固定数据集——
是对 init 的 trust region 而非真 PPO。已在代码中注释说明；如需教学上更接近
真 PPO，可每 epoch 把 reference 同步为上一轮 policy 快照。

## 4. 轻微备注（可不处理）

- _shared/restoration.py：clamp(±1) 对超范围输入梯度全零（输入契约为 [-1,1]）。
- _shared/pose_estimation.py："indices" 是平面 argmax 索引，局部变量名 keypoints 误导。
- co_segmentation/_common.py：softmax 后再除行和的无效除法、恒真 if 分支、
  双 transpose no-op。
- Llms 资源注册表固化了两个拼错的 PDF 文件名（"Chinchilia .pdf"、"mingpt4.pdf"）。
- GAN lesson label_smoothing 开启时 G 目标也被平滑（默认关闭不触发）。
- flow matching lesson_06 的 sample() 缺 @torch.no_grad()（仅浪费显存）。
