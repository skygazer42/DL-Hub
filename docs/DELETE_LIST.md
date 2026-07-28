# 待删除清单（实现完成后再删）

**规则（强制）：** 任何非 PDF 的旧实现，必须满足“新版本已重写 + 可运行 + 有验收（smoke/test）”后才允许删除。

> 说明：你已授权我“全权操作，可删除，不必逐次确认”。因此我会在本文件中持续记录删除决策与验收证据，保证过程可追溯。

> PDF（例如 `resources/pdfs/llms/` 下的论文）不在此清单范围内，不删除、不改内容。

## 状态说明

- **WIP**：新实现还没完成，旧代码不能删
- **Ready**：新实现已完成并通过验收，等待你确认后可删
- **Deleted**：已删除（仍可在 Git 历史中追溯）

## 清单（按轨道）

### Vision

| 旧路径（候选） | 新实现（替代） | 验收/证据（必须可复现） | 状态 | 备注 |
|---|---|---|---|---|
| `Deep_project/Mnist/LeNet/Mnist_LeNet.py` | `tracks/vision/lesson_01_mnist_lenet/` | `python scripts/smoke_check.py` | Deleted | 已删除（可在 Git 历史中追溯） |
| `Deep_project/Mnist/4.3实验报告/code/Mnist_LeNet.py` | `tracks/vision/lesson_01_mnist_lenet/` | `python scripts/smoke_check.py` | Deleted | 已删除（可在 Git 历史中追溯） |
| `Deep_project/Mnist/mlp/Mnist_mlp.py` | `tracks/vision/lesson_02_mnist_mlp/` | `python -m tracks.vision.lesson_02_mnist_mlp.train --dataset fake --epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --run-name smoke` | Deleted | 已删除（可在 Git 历史中追溯） |
| `Deep_project/Mnist/AlexNet/Mnist_AlexNet.py` | `tracks/vision/lesson_03_mnist_alexnet/` | `python -m tracks.vision.lesson_03_mnist_alexnet.train --dataset fake --resize-to 64 --epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --run-name smoke` | Deleted | 已删除（可在 Git 历史中追溯） |
| `Deep_project/Mnist/4.3实验报告/code/Mnist_AlexNet.py` | `tracks/vision/lesson_03_mnist_alexnet/` | `python -m tracks.vision.lesson_03_mnist_alexnet.train --dataset fake --resize-to 64 --epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --run-name smoke` | Deleted | 已删除（可在 Git 历史中追溯） |
| `Deep_project/Mnist/4.3实验报告/code/Mnist_mlp.py` | `tracks/vision/lesson_02_mnist_mlp/` | `python -m tracks.vision.lesson_02_mnist_mlp.train --dataset fake --epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --run-name smoke` | Deleted | 已删除（可在 Git 历史中追溯） |
| `Deep_project/FCOS_Pytorch_case/` | `tracks/vision/lesson_04_synthetic_detection_fcos/` | `python -m tracks.vision.lesson_04_synthetic_detection_fcos.train --device cpu --epochs 1 --num-samples 256 --batch-size 32 --max-train-batches 2 --max-eval-batches 2 --run-name smoke` | Deleted | 旧目录无 PDF；新版本用合成数据跑通 anchor-free 检测闭环 |
| `Deep_project/retiannet/` | `tracks/vision/lesson_04_synthetic_detection_fcos/` | `python scripts/smoke_check.py` | Deleted | 旧目录无 PDF；统一收敛到同一套检测实现 |
| `Deep_project/yolov5 run + qat+ teacher+purning/` | `tracks/vision/lesson_04_synthetic_detection_fcos/` | `python scripts/smoke_check.py` | Deleted | 旧目录无 PDF；统一收敛到同一套检测实现 |
| `transformer/`（ViT/T2T/TNT/零散实现） | `tracks/vision/lesson_05_vit_compact_classification/` + `tracks/vision/lesson_06_swin_compact_classification/` + `tracks/llm/lesson_01_compact_causal_lm_transformer/` | `python scripts/smoke_check.py` + `pytest -q tests/test_tracks_vision_transformers.py` | Deleted | 仅保留 PDF，已移动到 `resources/pdfs/transformers/` |
| `Deep_project/swin/` | `tracks/vision/lesson_06_swin_compact_classification/` | `python scripts/smoke_check.py` + `pytest -q tests/test_tracks_vision_transformers.py` | Deleted | 仅保留 PDF，已移动到 `resources/pdfs/transformers/` |
| `Deep_project/Mnist/`（报告/图片/LaTeX 等非 PDF） | `tracks/vision/lesson_01_mnist_lenet/` + `tracks/vision/lesson_02_mnist_mlp/` + `tracks/vision/lesson_03_mnist_alexnet/` | `python scripts/smoke_check.py` | Deleted | PDF 已移动到 `resources/pdfs/deep_project/Mnist/4.3实验报告/main.pdf` |
| `Deep_project/3D keypoint/`（mediapipe demo） | `tracks/vision/lesson_07_compact_keypoint_regression/` | `python scripts/smoke_check.py` + `pytest -q tests/test_tracks_vision_keypoints.py` | Deleted | 旧实现依赖 mediapipe/opencv，不再作为主线 |

### NLP

| 旧路径（候选） | 新实现（替代） | 验收/证据（必须可复现） | 状态 | 备注 |
|---|---|---|---|---|
| `Deep_project/ner/`（TF/Keras） | `tracks/nlp/lesson_03_compact_ner_bilstm/` | `python scripts/smoke_check.py` | Deleted | 已迁移为 PyTorch 版本并移除旧目录 |
| `Deep_project/reading_comprehension/Bruce-BiDAF/`（TF/Keras） | `tracks/nlp/lesson_07_reading_comprehension/` | `python scripts/smoke_check.py` | Deleted | 已迁移为 PyTorch compact span 版本并移除旧代码目录；PDF 已移动到 `resources/pdfs/deep_project/reading_comprehension/` |
| `Deep_project/reading_comprehension/09-机器阅读理解/Bruce-BiDAF/` | `tracks/nlp/lesson_07_reading_comprehension/` | `python scripts/smoke_check.py` | Deleted | PDF 已移动到 `resources/pdfs/deep_project/reading_comprehension/` |
| `Deep_project/Text generation tf/`（TF/Keras） | `tracks/nlp/lesson_04_compact_seq2seq_attention_generation/` | `python scripts/smoke_check.py` | Deleted | 旧目录含 zip/IDE 文件；新版本用 compact 数据实现 Seq2Seq + Bahdanau Attention，并输出 `samples.jsonl` |
| `Deep_project/keras_text_classification/`（TF/Keras） | `tracks/nlp/lesson_05_compact_text_classification_textcnn/` + `tracks/nlp/lesson_06_compact_text_classification_bilstm/` | `python scripts/smoke_check.py` | Deleted | 旧目录含 `.h5` 与 IDE 文件；新版本统一到 compact 数据集与统一训练脚手架 |
| `Deep_project/ERnerclassification/`（pytorch_pretrained + Bert/Ernie） | `tracks/llm/lesson_01_compact_causal_lm_transformer/` + `tracks/nlp/*`（统一训练脚手架） | `python scripts/smoke_check.py` | Deleted | 旧目录依赖过时库；新版本把 “token 级训练闭环 + 生成” 收敛到 LLM 轨，并保留可复现验收命令 |
| `Deep_project/datapreprocessing for nlp/`（旧预处理脚本） | `tracks/nlp/synthetic_text.py` + `tracks/nlp/synthetic_seq2seq.py` + `tracks/llm/synthetic_lm.py` | `python scripts/smoke_check.py` | Deleted | 旧脚本不再作为主入口；新版本用统一的 dataloader/配置风格覆盖所需功能 |

### GNN

| 旧路径（候选） | 新实现（替代） | 验收/证据（必须可复现） | 状态 | 备注 |
|---|---|---|---|---|
| `graph/pygcn/pygcn改/` | `tracks/gnn/lesson_04_cora_node_classification_gcn/` | `python scripts/smoke_check.py` | Deleted | 已迁移为纯 PyTorch 最小实现并移除旧目录（Git 历史可追溯）；Cora 数据移动到 `tracks/gnn/assets/cora/` |
| `graph/GAT/pyGAT-master-jupyter/` | `tracks/gnn/lesson_03_gat_compact_graph_classification/` | `python scripts/smoke_check.py` | Deleted | 已迁移为纯 PyTorch 最小实现并移除旧目录（Git 历史可追溯） |
| `graph/gin/` | `tracks/gnn/lesson_02_gin_compact_graph_classification/` | `python scripts/smoke_check.py` | Deleted | 已迁移为纯 PyTorch 最小实现并移除旧目录（Git 历史可追溯） |
| `graph/label_propagation/`（DGL） | `tracks/gnn/lesson_05_label_propagation_cora/` | `python scripts/smoke_check.py` | Deleted | 已迁移为纯 PyTorch sparse 最小实现并移除旧目录 |
| `graph/graphsage/`（数据复制 + IDE 文件） | `tracks/gnn/lesson_06_graphsage_cora/` | `python scripts/smoke_check.py` | Deleted | 已由 GraphSAGE 最小实现替代并移除旧目录 |
| `graph/SDNE/` | `tracks/gnn/lesson_07_sdne_karate_embedding/` | `python scripts/smoke_check.py` | Deleted | 已迁移为 SDNE 风格最小实现并移除旧目录（Karate 数据放在 `tracks/gnn/assets/karate/`） |
| `graph/LINE/` | `tracks/gnn/lesson_08_line_karate_embedding/` | `python scripts/smoke_check.py` | Deleted | 已迁移为 LINE 风格最小实现并移除旧目录 |
| `graph/metapath2vec/` | `tracks/gnn/lesson_09_metapath2vec_compact_hetero_embedding/` | `python -m tracks.gnn.lesson_09_metapath2vec_compact_hetero_embedding.train --device cpu --epochs 1 --num-walks 50 --walk-length 10 --window-size 3 --run-name smoke` | Deleted | 旧目录已移除；新版本使用 compact 异构图 + metapath 随机游走 |
| `graph/pinsage/`（DGL 示例工程） | `tracks/gnn/lesson_10_pinsage_compact_recommender/` | `python -m tracks.gnn.lesson_10_pinsage_compact_recommender.train --device cpu --epochs 1 --steps-per-epoch 5 --batch-size 64 --num-users 64 --num-items 128 --eval-k 10 --run-name smoke` | Deleted | 旧目录含 IDE 文件与外部数据依赖；新版本改为纯 PyTorch compact 推荐图 |
| `graph/Rgcn/`（DGL + aifb） | `tracks/gnn/lesson_11_rgcn_compact_node_classification/` | `python -m tracks.gnn.lesson_11_rgcn_compact_node_classification.train --device cpu --epochs 2 --num-nodes 120 --edges-per-node 2 --run-name smoke` | Deleted | 旧目录依赖过时 DGL API；新版本实现 R-GCN + basis 分解（compact 关系图） |

### PointCloud

| 旧路径（候选） | 新实现（替代） | 验收/证据（必须可复现） | 状态 | 备注 |
|---|---|---|---|---|
| `Deep_project/Pointnet_Pointnet2/` | `tracks/pointcloud/lesson_01_pointnet_compact_classification/` + `tracks/pointcloud/lesson_03_pointnet2_compact_classification/` | `python scripts/smoke_check.py` | Deleted | 已由 PointNet / PointNet2 最小实现替代并移除旧目录 |
| `Deep_project/dgcnn/` | `tracks/pointcloud/lesson_02_dgcnn_compact_classification/` | `python scripts/smoke_check.py` | Deleted | 已由 DGCNN 最小实现替代并移除旧目录 |

### Generative

| 旧路径（候选） | 新实现（替代） | 验收/证据（必须可复现） | 状态 | 备注 |
|---|---|---|---|---|
| `Deep_project/VAE/` | `tracks/generative/lesson_01_vae_mnist/` | `python scripts/smoke_check.py` | Deleted | 已由统一 VAE lesson 替代并移除旧目录（Git 历史可追溯） |
| `GAN/` | `tracks/generative/lesson_02_gan_mnist/` | `python scripts/smoke_check.py` | Deleted | 旧目录已移动到 `resources/pdfs/gan/`（PDF/笔记保留） |
