# Deep Learning Repository

此仓库收录了机器学习、深度学习、计算机视觉、自然语言处理和爬虫等方向的代码示例与学习资料，便于快速检索和复现。

## 项目导航
- **Clip**：论文解析与核心概念整理。
- **Deep Projects**：
  - 3D Keypoint：3D 关键点检测
  - Data Preprocessing for NLP：NLP 数据预处理
  - DGCNN：动态图卷积神经网络
  - ERNerClassification：实体识别与分类
  - FCOS_Pytorch_Case：FCOS 目标检测 Pytorch 实现
  - Fraud Prediction：欺诈预测
  - Keras Text Classification：Keras 文本分类
  - Lebert NER：基于 Lebert 的命名实体识别
  - Mnist：LeNet、AlexNet、GoogLeNet 等模型实现
  - NER：命名实体识别
  - Pointnet/Pointnet2：点云网络
  - Reading Comprehension：阅读理解
  - RetinaNet：单阶段目标检测
  - Swin：Swin Transformer
  - T5：Text-to-Text Transfer Transformer
  - Text Generation TF：TensorFlow 文本生成
  - VAE：变分自编码器
  - YOLOv5：目标检测 + 量化感知训练 + 教师模型 + 剪枝
- **Deep Learning Notes**：计算机视觉相关笔记。
- **GAN**：WGAN、Improved GAN 等生成对抗网络实现。
- **Graph Replications**：GAT、GIN、GraphSAGE、Label Propagation、LINE、Metapath2Vec、MPNN、PinSAGE、PyGCN、RGCN、SDNE 等图模型复现。
- **LLMs**：大模型相关的 50 余篇资料。
- **Machine Learning Algorithms**：欺诈检测等传统机器学习算法。
- **ONNX**：Sklearn ONNX 转换、ONNX Runtime、PNNX 等工具示例。
- **Optimization**：Matlab 优化实现。
- **Spider**：Requests 与 Scrapy 示例。
- **SQL**：SQL 学习笔记。
- **Transformer**：Transformer 模型解读。
- **LangChain**：LangChain 相关资料。

## 示例：在 MNIST 上训练 LeNet
位于 `Deep_project/Mnist/LeNet/Mnist_LeNet.py` 的脚本提供了可配置的训练与评估流程，支持 GPU/CPU 自动切换，并允许通过命令行快速做小批量验证。

### 依赖
- Python 3.10+
- PyTorch、TorchVision
- Matplotlib、NumPy

使用 pip 安装核心依赖：

```bash
pip install torch torchvision matplotlib numpy
```

### 快速运行
下载数据并训练 1 个 epoch，同时限制训练/验证批次数以快速验证流程：

```bash
python Deep_project/Mnist/LeNet/Mnist_LeNet.py \
  --epochs 1 \
  --max-train-batches 5 \
  --max-eval-batches 2
```

脚本会输出每轮准确率，并在运行结束后保存：
- 训练曲线：`lenet_accuracy.png`
- 模型权重：`LeNet.pth`

## 反馈
欢迎通过 Issue 或 Pull Request 提出建议与改进，让这些资源更好地服务学习与研究。
