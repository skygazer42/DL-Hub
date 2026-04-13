<div align="center">

<img src="assets/hero_banner.png" width="100%" alt="DL-Hub 鈥?Deep Learning from Scratch" />

# DL-Hub

**浠庨浂鎵嬪啓锛屽惊搴忔笎杩?鈥?PyTorch 娣卞害瀛︿範缁熶竴瀛︿範椤圭洰**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

<code>76 Lessons</code> 路 <code>8 Learning Tracks</code> 路 <code>27 ML Algorithms</code> 路 <code>2500+ Model Zoo Architectures</code> 路 <code>126 Test Files</code>

<br/>

缁熶竴浠ｇ爜椋庢牸銆佺粺涓€璁粌鑴氭墜鏋躲€佺粺涓€杩愯鏂瑰紡<br/>
璁╁涔犺€呯湡姝ｈ兘 **"寰簭娓愯繘璺戦€?鈫?鏀瑰緱鍔?鈫?鑳介獙鏀?**

[Quick Start](#-quick-start) 路 [Learning Tracks](#-learning-tracks) 路 [Model Zoo](#-model-zoo) 路 [Federated Zoo](#-federated-learning-zoo) 路 [ML Algorithms](#-numpy-ml-algorithms) 路 [Docs](#-documentation)

</div>

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝崄鍥涳級
> Continue the unseen-direction expansion with a cross-domain batch that covers relighting, transparent segmentation, event vision, point cloud motion/completion, multimodal reasoning, and text-conditioned 3D generation. Each direction again lands 10 toy-first families.

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Image Relighting / 鍥惧儚閲嶇収鏄?| 10 | `dlhub/vision/image_relighting/` |
| Transparent Object Segmentation / 閫忔槑鐗╀綋鍒嗗壊 | 10 | `dlhub/vision/transparent_object_segmentation/` |
| Video Matting / 瑙嗛鎶犲儚 | 10 | `dlhub/vision/video_matting/` |
| Event Camera Understanding / 浜嬩欢鐩告満鐞嗚В | 10 | `dlhub/vision/event_camera_understanding/` |
| Scene Flow / 鍦烘櫙娴?| 10 | `dlhub/pointcloud/scene_flow/` |
| Point Cloud Completion / 鐐逛簯琛ュ叏 | 10 | `dlhub/pointcloud/pointcloud_completion/` |
| Audio-Visual Learning / 闊宠瀛︿範 | 10 | `dlhub/multimodal/audio_visual_learning/` |
| Multimodal Reasoning / 澶氭ā鎬佹帹鐞?| 10 | `dlhub/multimodal/multimodal_reasoning/` |
| Video Diffusion / 瑙嗛鎵╂暎 | 10 | `dlhub/generative/video_diffusion/` |
| Text-to-3D / 鏂囨湰鐢熸垚涓夌淮 | 10 | `dlhub/generative/text_to_3d/` |


#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝崄浜旓級
> Continue the unseen-direction expansion with a cross-domain batch focused on deraining, shadow analysis, 3D upsampling/correspondence/open-vocabulary modeling, multimodal retrieval/navigation/document VLMs, and image-conditioned video/3D generation. Each direction again lands 10 toy-first families.
| 閺傜懓鎮?| 瑜版挸澧犵€硅埖妫岄弫?| 閸栧懓鐭惧?|
|------|-----------|--------|
| Image Deraining / 閸ユ儳鍎氶崢濠氭处 | 10 | `dlhub/vision/image_deraining/` |
| Shadow Detection / 闂冩潙濂栧Λ鈧ù? | 10 | `dlhub/vision/shadow_detection/` |
| Point Cloud Upsampling / 閻愰€涚隘娑撳﹪鍣伴弽? | 10 | `dlhub/pointcloud/pointcloud_upsampling/` |
| Shape Correspondence 3D / 娑撳娣ぐ銏㈠Ц鐎电懓绨?| 10 | `dlhub/pointcloud/shape_correspondence_3d/` |
| Open Vocabulary 3D / 瀵偓閺€鎹愮槤鐞涖劋绗佺紒? | 10 | `dlhub/pointcloud/open_vocabulary_3d/` |
| Image-Text Retrieval / 閸ョ偓鏋冨Λ鈧槐? | 10 | `dlhub/multimodal/image_text_retrieval/` |
| Vision-Language Navigation / 鐟欏棜顫庣拠顓♀枅鐎佃壈鍩?| 10 | `dlhub/multimodal/vision_language_navigation/` |
| Document VLM / 閺傚洦銆?VLM | 10 | `dlhub/multimodal/document_vlm/` |
| Image-to-Video / 閸ュ墽鏁撶憴鍡涱暥 | 10 | `dlhub/generative/image_to_video/` |
| Image-to-3D / 閸ュ墽鏁撴稉澶屾樊 | 10 | `dlhub/generative/image_to_3d/` |

---

## What You'll Build

<table>
<tr>
<td align="center" width="25%">
<br/>
<b>Vision</b><br/>
<sub>浠?LeNet 鍒?ViT锛?br/>736 鏋舵瀯 路 鍥惧儚鍒嗙被 / 妫€娴?/ 鍒嗗壊</sub>
</td>
<td align="center" width="25%">
<br/>
<b>NLP</b><br/>
<sub>浠庤瘝宓屽叆鍒?Transformer锛?br/>813 鏋舵瀯 路 鍒嗙被 / NER / 闃呰鐞嗚В</sub>
</td>
<td align="center" width="25%">
<br/>
<b>GNN</b><br/>
<sub>浠?GCN 鍒?PinSAGE锛?br/>鍥惧垎绫?/ 鑺傜偣宓屽叆 / 鎺ㄨ崘</sub>
</td>
<td align="center" width="25%">
<br/>
<b>Point Cloud</b><br/>
<sub>浠?PointNet 鍒?PCT锛?br/>64 鏋舵瀯 路 鍒嗙被 / 閮ㄤ欢鍒嗗壊 / 閲嶅缓 / 15 绉嶈嚜鐩戠潱</sub>
</td>
</tr>
<tr>
<td align="center" width="25%">
<br/>
<b>Generative</b><br/>
<sub>VAE & GAN锛?br/>鎵嬪啓鏁板瓧閲嶅缓涓庣敓鎴?/sub>
</td>
<td align="center" width="25%">
<br/>
<b>Multimodal</b><br/>
<sub>浠?CLIP 鍒?LLaVA锛?0 VLM 鏋舵瀯<br/>瑙嗚闂瓟 / 鐩爣妫€娴?/ 鏃跺簭瀹氫綅</sub>
</td>
<td align="center" width="25%">
<br/>
<b>LLM</b><br/>
<sub>Causal LM + 璧勬簮搴擄紝<br/>50+ 璁烘枃绗旇</sub>
</td>
<td align="center" width="25%">
<br/>
<b>Federated</b><br/>
<sub>76 鑱旈偊绛栫暐<br/>宸垎闅愮 / 瀹夊叏鑱氬悎 / 涓€у寲</sub>
</td>
</tr>
</table>

<p align="center">
  <img src="assets/overview_8panels.png" width="80%" alt="DL-Hub 鍏ぇ棰嗗煙锛歏ision 路 NLP 路 GNN 路 Point Cloud 路 Generative 路 Multimodal 路 LLM 路 Federated" />
</p>
<p align="center"><sub>鈶?Vision 鈥?CNN / ViT 鍥惧儚鍒嗙被 路 鈶?NLP 鈥?鏂囨湰鍒嗙被 / NER 路 鈶?GNN 鈥?鍥剧缁忕綉缁?路 鈶?Point Cloud 鈥?3D 鐐逛簯 路 鈶?Generative 鈥?VAE / GAN 路 鈶?Multimodal 鈥?VLM 瑙嗚璇█ 路 鈶?LLM 鈥?澶ц瑷€妯″瀷 路 鈶?Federated 鈥?鑱旈偊瀛︿範</sub></p>

---

## Contents

- [What You'll Build](#what-youll-build)
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Learning Path](#-learning-path)
- [Learning Tracks](#-learning-tracks)
  - [Foundations](#-foundations--鍩虹) 路 [Vision](#-vision--瑙嗚) 路 [NLP](#-nlp--鑷劧璇█澶勭悊) 路 [GNN](#-gnn--鍥剧缁忕綉缁? 路 [Point Cloud](#-point-cloud--鐐逛簯) 路 [Generative](#-generative--鐢熸垚妯″瀷) 路 [LLM](#-llm--澶ц瑷€妯″瀷) 路 [Multimodal](#-multimodal--澶氭ā鎬?
- [Model Zoo](#-model-zoo)
  - [Vision Zoo (736 architectures)](#vision-zoo--736-architectures) 路 [NLP Zoo (813 architectures)](#nlp-zoo--813-architectures) 路 [Point Cloud Zoo (64 architectures)](#point-cloud-zoo--64-architectures) 路 [VLM Zoo (70 families)](#vlm-zoo--70-families) 路 [Generative Zoo (GAN + Diffusion)](#generative-zoo--gan--diffusion)
- [Federated Learning Zoo](#-federated-learning-zoo)
- [NumPy ML Algorithms](#-numpy-ml-algorithms)
- [Optimization Toolkit](#-optimization-toolkit)
- [Documentation](#-documentation)
- [Design Philosophy](#-design-philosophy)
- [Contributing](#-contributing)
- [Citation](#-citation)

---

## Quick Start

> [!TIP]
> 鎵€鏈?lesson 鍧囨敮鎸?`--dataset fake` 绂荤嚎鍐掔儫 鈥?**鏃犻渶涓嬭浇浠讳綍鏁版嵁闆嗭紝2 鍒嗛挓鍗冲彲璺戦€?*銆?

```bash
# 鍏嬮殕浠撳簱
git clone https://github.com/skygazer42/DL-Hub.git
cd DL-Hub
pip install -r requirements.txt

# 浠撳簱绾у啋鐑熸祴璇曪紙楠岃瘉鐜锛?
python scripts/smoke_check.py

# 璺戦€氱涓€涓?lesson
python -m tracks.vision.lesson_01_mnist_lenet.train \
  --dataset fake --epochs 1 \
  --max-train-batches 2 --max-eval-batches 2
```

**鍒楀嚭鎵€鏈夊彲杩愯鐨?lesson**锛?

```bash
python scripts/run_lesson.py --list
```

<details>
<summary><b>缁熶竴 CLI 鍙傛暟锛堟墍鏈?lesson 閫氱敤锛?/b></summary>

| 鍙傛暟 | 璇存槑 | 绀轰緥 |
|------|------|------|
| `--dataset` | 鏁版嵁妯″紡 | `fake` (绂荤嚎鍐掔儫) / `toy` / `real` |
| `--epochs` | 璁粌杞暟 | `10` |
| `--batch-size` | 鎵瑰ぇ灏?| `32` |
| `--learning-rate` | 瀛︿範鐜?| `0.001` |
| `--seed` | 闅忔満绉嶅瓙 | `42` |
| `--device` | 璁＄畻璁惧 | `cpu` / `cuda` / `mps` / `auto` |
| `--max-train-batches` | 闄愬埗璁粌 batch 鏁?| `2` |
| `--max-eval-batches` | 闄愬埗璇勪及 batch 鏁?| `2` |

</details>

---

## Prerequisites

> [!NOTE]
> 鏈」鐩€傚悎鏈変竴瀹?Python 鍩虹鐨勫涔犺€呫€備互涓嬫槸鍚?track 鐨勫厛淇缓璁€?

| Track | 鍏堜慨鐭ヨ瘑 |
|-------|---------|
| Foundations | Python 鍩虹銆佺嚎鎬т唬鏁板叆闂?|
| Vision | Foundations track + 鍗风Н鐩磋 |
| NLP | Foundations track + 鏂囨湰澶勭悊鍩虹 |
| GNN | Foundations track + 鍥捐鍩烘湰姒傚康 |
| Point Cloud | Vision track + 3D 鍑犱綍鐩磋 |
| Generative | Vision track + 姒傜巼璁哄熀纭€ |
| LLM | NLP track + Transformer 鏈哄埗 |
| Multimodal | Vision track + NLP track + 娉ㄦ剰鍔涙満鍒?|

---

## Learning Path

涓嶇煡閬撲粠鍝紑濮嬶紵鏍规嵁浣犵殑鏃堕棿閫夋嫨涓€鏉″涔犺矾绾匡細

<p align="center">
  <img src="assets/learning_path_steps.png" width="85%" alt="8 Learning Tracks: Foundations 鈫?Vision 鈫?NLP 鈫?GNN 鈫?Point Cloud 鈫?Generative 鈫?LLM 鈫?Multimodal" />
</p>
<p align="center"><sub>Step 1鈥? 瀵瑰簲锛欶oundations 鈫?Vision 鈫?NLP 鈫?GNN 鈫?Point Cloud 鈫?Generative 鈫?LLM 鈫?Multimodal</sub></p>

<table>
<tr>
<th width="20%">璺嚎</th>
<th width="15%">鏃堕棿</th>
<th width="15%">Lessons</th>
<th width="50%">鍐呭</th>
</tr>
<tr>
<td><b>Weekend Sprint</b></td>
<td>1-2 澶?/td>
<td>6 lessons</td>
<td>Foundations (2) 鈫?Vision lesson 01-02 鈫?Generative lesson 01 鈫?LLM lesson 01<br/><sub>蹇€熷缓绔嬩粠寮犻噺鍒扮敓鎴愭ā鍨嬬殑瀹屾暣鐩磋</sub></td>
</tr>
<tr>
<td><b>Two-Week Deep Dive</b></td>
<td>2 鍛?/td>
<td>18 lessons</td>
<td>Foundations (2) 鈫?Vision (5) 鈫?NLP (4) 鈫?GNN (3) 鈫?Generative (2) 鈫?LLM (1) 鈫?Point Cloud (1)<br/><sub>瑕嗙洊鎵€鏈?track 鐨勬牳蹇?lesson</sub></td>
</tr>
<tr>
<td><b>Full Curriculum</b></td>
<td>6-8 鍛?/td>
<td>76 lessons</td>
<td>鎸夐『搴忓畬鎴愬叏閮?8 涓?track 鐨勬墍鏈?lesson<br/><sub>绯荤粺鎺屾彙浠庣粡鍏?ML 鍒板墠娌挎繁搴﹀涔犵殑瀹屾暣鎶€鑳芥爲</sub></td>
</tr>
</table>

> [!TIP]
> 鎺ㄨ崘椤哄簭锛?*Foundations 鈫?Vision 鈫?NLP 鈫?GNN 鈫?Point Cloud 鈫?Generative 鈫?LLM 鈫?Multimodal**銆傛瘡涓?lesson 閮芥湁鐙珛鐨?README 璇存槑鐩爣銆佸厛淇拰楠屾敹鏍囧噯銆?

---

## 璇剧▼鍙婁唬鐮佸悎闆?

<table>
<tr>
<td align="center" width="12%"><b>Foundations</b><br/><sub>2 lessons</sub></td>
<td align="center" width="12%"><b>Vision</b><br/><sub>14 lessons</sub></td>
<td align="center" width="12%"><b>NLP</b><br/><sub>7 lessons</sub></td>
<td align="center" width="12%"><b>GNN</b><br/><sub>11 lessons</sub></td>
<td align="center" width="12%"><b>Point Cloud</b><br/><sub>23 lessons</sub></td>
<td align="center" width="12%"><b>Generative</b><br/><sub>2 lessons</sub></td>
<td align="center" width="12%"><b>LLM</b><br/><sub>1 lesson</sub></td>
<td align="center" width="12%"><b>Multimodal</b><br/><sub>16 lessons</sub></td>
</tr>
</table>

---

### 鈿?1. Foundations / 鍩虹

> 寮犻噺銆佽嚜鍔ㄦ眰瀵笺€佽缁冨惊鐜叆闂?鈥?鎵€鏈夊悗缁?track 鐨勫熀鐭炽€?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | 寮犻噺鎿嶄綔 & Autograd 鏈哄埗 | [lesson_01_tensors](tracks/foundations/lesson_01_tensors/) | `torch.Tensor`, `backward()`, 璁＄畻鍥?|
| 2 | 浠庨浂瀹炵幇绾挎€у洖褰?| [lesson_02_linear_regression](tracks/foundations/lesson_02_linear_regression_autograd/) | 姊害涓嬮檷, 鎹熷け鍑芥暟, 鍙傛暟鏇存柊 |

---

### 馃憗锔?2. Vision / 瑙嗚

> 浠?MNIST 鍏ラ棬鍒扮洰鏍囨娴嬨€佽涔夊垎鍓层€乂ision Transformer銆?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | LeNet-5 鍥惧儚鍒嗙被 | [mnist_lenet](tracks/vision/lesson_01_mnist_lenet/) | 鍗风Н灞? 姹犲寲, 鍏ㄨ繛鎺?|
| 2 | MLP 鍥惧儚鍒嗙被 | [mnist_mlp](tracks/vision/lesson_02_mnist_mlp/) | 澶氬眰鎰熺煡鏈? Flatten |
| 3 | AlexNet 鍥惧儚鍒嗙被 | [mnist_alexnet](tracks/vision/lesson_03_mnist_alexnet/) | 娣卞眰鍗风Н缃戠粶, Dropout |
| 4 | FCOS 鐩爣妫€娴?| [synthetic_detection_fcos](tracks/vision/lesson_04_synthetic_detection_fcos/) | Anchor-free, FPN, 鍥炲綊澶?|
| 5 | ViT 鍥惧儚鍒嗙被 | [vit_toy_classification](tracks/vision/lesson_05_vit_toy_classification/) | Patch Embedding, Self-Attention |
| 6 | Swin Transformer 鍥惧儚鍒嗙被 | [swin_toy_classification](tracks/vision/lesson_06_swin_toy_classification/) | Window Attention, Shifted Window |
| 7 | 鍏抽敭鐐瑰洖褰?| [toy_keypoint_regression](tracks/vision/lesson_07_toy_keypoint_regression/) | 鍧愭爣鍥炲綊, Heatmap |
| 8 | UNet 璇箟鍒嗗壊 | [synthetic_segmentation_unet](tracks/vision/lesson_08_synthetic_segmentation_unet/) | Encoder-Decoder, Skip Connection |
| 9 | 澶?Backbone 瀵规瘮 | [cnn_backbones_toy_classification](tracks/vision/lesson_09_cnn_backbones_toy_classification/) | 缁熶竴鎺ュ彛, 鐗瑰緛鎻愬彇 |
| 10 | 鍥惧儚鍘诲櫔锛堝妯″瀷锛?| [synthetic_denoising](tracks/vision/lesson_10_synthetic_denoising/) | 鍚堟垚鍣０寤烘ā, 鍘诲櫔鍥炲綊 |
| 11 | YOLACT 瀹炰緥鍒嗗壊 | [synthetic_instance_segmentation_yolact](tracks/vision/lesson_11_synthetic_instance_segmentation_yolact/) | Prototype + Coefficients |
| 12 | YOLO 椋庢牸鐩爣妫€娴?| [synthetic_detection_yolo](tracks/vision/lesson_12_synthetic_detection_yolo/) | Grid/Objectness + BBox |
| 13 | 琛屼汉妫€娴嬶紙FCOS锛?| [synthetic_pedestrian_detection_fcos](tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/) | Anchor-free 妫€娴嬪ご |
| 14 | 瑙嗛澶氱洰鏍囪窡韪紙MOT锛?| [video_mot_basics](tracks/vision/lesson_14_video_mot_basics/) | 澶氱洰鏍囪建杩归娴? Presence + IoU |

<details>
<summary><b>鏀寔鐨?Vision Backbones锛?08 绠楁硶鏃?/ 736 鏋舵瀯 ID锛?/b></summary>

| 绫诲埆 | 浠ｈ〃鏋舵瀯 |
|------|---------|
| 缁忓吀 CNN | AlexNet, VGG, GoogLeNet, ResNet, DenseNet, SqueezeNet |
| 楂樻晥缃戠粶 | MobileNet v1-v4, EfficientNet, GhostNet v1/v2, ShuffleNet, MNASNet, FBNet, MicroNet |
| 娉ㄦ剰鍔?CNN | SENet, CBAM, BAM, ECA-Net, SK-Net, CoordAtt, SimAM, Triplet Attention |
| 鐜颁唬 CNN | ConvNeXt v1/v2, RepVGG, RepLKNet, InceptionNeXt, HorNet, FocalNet, SLaK |
| Vision Transformer | ViT, DeiT, DeiT3, BEiT, EVA, CaiT, CrossViT, Swin v2, CSwin, MAE-ViT |
| 楂樻晥 Transformer | EfficientViT, TinyViT, EdgeViT, LightViT, FastViT, FasterViT, SwiftFormer |
| MLP 绯诲垪 | MLP-Mixer, gMLP, ResMLP, FNet, CycleMLP, AS-MLP, WaveMLP, MorphMLP |
| Hybrid | CoAtNet, MobileFormer, ConvFormer, Uniformer, CMT, MaxViT, MobileViT v1-v3 |
| 鐗规畩缁撴瀯 | CapsNet, ScatterNet, FractalNet, HighwayNet, HRNet, NAS 绯诲垪 |

> 瀹屾暣鍒楄〃瑙?`python -m dlhub.vision.backbones.catalog --list`锛屾墍鏈?backbone 鍧囦负绾?PyTorch 鏈湴瀹炵幇銆?

</details>

---

### 馃摑 3. NLP / 鑷劧璇█澶勭悊

> 浠?toy 鏂囨湰鍒嗙被鍒?Transformer銆丯ER銆侀槄璇荤悊瑙ｃ€?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | Embedding + FC 鏂囨湰鍒嗙被 | [toy_text_classification](tracks/nlp/lesson_01_toy_text_classification/) | 璇嶅祵鍏? 璇嶈 |
| 2 | Transformer Encoder 鏂囨湰鍒嗙被 | [toy_text_classification_transformer](tracks/nlp/lesson_02_toy_text_classification_transformer/) | Self-Attention, 浣嶇疆缂栫爜 |
| 3 | BiLSTM 鍛藉悕瀹炰綋璇嗗埆 | [toy_ner_bilstm](tracks/nlp/lesson_03_toy_ner_bilstm/) | 搴忓垪鏍囨敞, BIO 鏍囩 |
| 4 | Seq2Seq + Attention 搴忓垪鐢熸垚 | [toy_seq2seq_attention_generation](tracks/nlp/lesson_04_toy_seq2seq_attention_generation/) | Encoder-Decoder, Bahdanau Attention |
| 5 | TextCNN 鏂囨湰鍒嗙被 | [toy_text_classification_textcnn](tracks/nlp/lesson_05_toy_text_classification_textcnn/) | 澶氬昂搴﹀嵎绉牳, 鏂囨湰鐗瑰緛 |
| 6 | BiLSTM 鏂囨湰鍒嗙被 | [toy_text_classification_bilstm](tracks/nlp/lesson_06_toy_text_classification_bilstm/) | 鍙屽悜 LSTM, 闅愯棌鐘舵€?|
| 7 | Span Prediction 闃呰鐞嗚В | [reading_comprehension](tracks/nlp/lesson_07_reading_comprehension/) | SQuAD 椋庢牸, Start/End Logits |

---

### 馃暩锔?4. GNN / 鍥剧缁忕綉缁?

> 鏈€涓板瘜鐨?track 鈥?浠?toy 鍥惧垎绫诲埌 Cora 鑺傜偣鍒嗙被銆佸浘宓屽叆銆佸紓鏋勫浘鎺ㄨ崘銆?

**Graph Classification**

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | GCN 鍥惧垎绫?| [toy_graph_classification](tracks/gnn/lesson_01_toy_graph_classification/) | 閭绘帴鐭╅樀, 娑堟伅浼犻€?|
| 2 | GIN 鍥惧垎绫?| [gin_toy_graph_classification](tracks/gnn/lesson_02_gin_toy_graph_classification/) | WL Test, 鍥惧悓鏋?|
| 3 | GAT 鍥惧垎绫?| [gat_toy_graph_classification](tracks/gnn/lesson_03_gat_toy_graph_classification/) | 娉ㄦ剰鍔涚郴鏁? 澶氬ご娉ㄦ剰鍔?|

**Node Classification**

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 4 | GCN Cora 鑺傜偣鍒嗙被 | [cora_node_classification_gcn](tracks/gnn/lesson_04_cora_node_classification_gcn/) | 鍗婄洃鐫ｅ涔? 璋辨柟娉?|
| 5 | Label Propagation Cora | [label_propagation_cora](tracks/gnn/lesson_05_label_propagation_cora/) | 缁忓吀鍩虹嚎, 鏃犲弬鏁版柟娉?|
| 6 | GraphSAGE Cora | [graphsage_cora](tracks/gnn/lesson_06_graphsage_cora/) | 閲囨牱鑱氬悎, 褰掔撼瀛︿範 |

**Embedding & Advanced**

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 7 | SDNE 鑺傜偣宓屽叆 | [sdne_karate_embedding](tracks/gnn/lesson_07_sdne_karate_embedding/) | 鑷紪鐮佸櫒, 涓€闃?浜岄樁杩戜技 |
| 8 | LINE 鑺傜偣宓屽叆 | [line_karate_embedding](tracks/gnn/lesson_08_line_karate_embedding/) | 澶ц妯＄綉缁? 杈归噰鏍?|
| 9 | Metapath2Vec 寮傛瀯鍥惧祵鍏?| [metapath2vec_toy_hetero_embedding](tracks/gnn/lesson_09_metapath2vec_toy_hetero_embedding/) | 鍏冭矾寰? 寮傛瀯闅忔満娓歌蛋 |
| 10 | PinSAGE 鎺ㄨ崘 | [pinsage_toy_recommender](tracks/gnn/lesson_10_pinsage_toy_recommender/) | 闅忔満娓歌蛋閲囨牱, 宸ヤ笟绾у浘鎺ㄨ崘 |
| 11 | R-GCN 鍏崇郴鍥捐妭鐐瑰垎绫?| [rgcn_toy_node_classification](tracks/gnn/lesson_11_rgcn_toy_node_classification/) | 鍏崇郴鐗瑰畾鏉冮噸, 鐭ヨ瘑鍥捐氨 |

---

### 鈽侊笍 5. Point Cloud / 鐐逛簯

> 3D 鐐逛簯鍒嗙被锛歅ointNet 鈫?DGCNN 鈫?PointNet++ 鈫?30+ Backbone Zoo銆?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | PointNet 鐐逛簯鍒嗙被 | [pointnet_toy_classification](tracks/pointcloud/lesson_01_pointnet_toy_classification/) | 鐐归泦鎺掑垪涓嶅彉鎬? T-Net |
| 2 | DGCNN 鐐逛簯鍒嗙被 | [dgcnn_toy_classification](tracks/pointcloud/lesson_02_dgcnn_toy_classification/) | 鍔ㄦ€佸浘, EdgeConv |
| 3 | PointNet++ 鐐逛簯鍒嗙被 | [pointnet2_toy_classification](tracks/pointcloud/lesson_03_pointnet2_toy_classification/) | 灞傜骇閲囨牱, Set Abstraction |
| 4 | 30+ Backbone Zoo 瀵规瘮 | [pointcloud_zoo_toy_classification](tracks/pointcloud/lesson_04_pointcloud_zoo_toy_classification/) | 缁熶竴鎺ュ彛, Backbone 瀵规瘮 |

<details>
<summary><b>鏀寔鐨?Point Cloud Backbones锛?0 绠楁硶 / 64 鏋舵瀯 ID锛?/b></summary>

| 绫诲埆 | 鏋舵瀯 |
|------|------|
| Set Models | PointNet, PointNet++, DeepSets |
| Graph Models | DGCNN, PointGAT, PointGCN, PointWeb |
| MLP Models | PointMLP, PointMixer, PointNeXt |
| Transformer | PCT, Point Transformer, PointBERT, PointMAE |
| Conv Models | KPConv, PointCNN, PointConv, ShellNet |
| Extra | CurveNet, GDANet, PAConv, PVCNN, RandLANet, RSCNN, SpiderCNN 绛?|

</details>

---

### 馃帹 6. Generative / 鐢熸垚妯″瀷

> VAE & GAN 鏈€灏忓疄鐜?鈥?鏀寔 `--dataset fake` 绂荤嚎鍐掔儫銆?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | VAE 閲嶅缓 & 鐢熸垚 | [vae_mnist](tracks/generative/lesson_01_vae_mnist/) | 閲嶅弬鏁板寲鎶€宸? KL 鏁ｅ害, ELBO |
| 2 | GAN 鐢熸垚 | [gan_mnist](tracks/generative/lesson_02_gan_mnist/) | 鐢熸垚鍣?鍒ゅ埆鍣ㄥ鎶? 绾充粈鍧囪　 |

---

### 馃 7. LLM / 澶ц瑷€妯″瀷

> Toy Causal Language Model 鈥?浠庨浂鎼缓 Transformer 鐢熸垚妯″瀷銆?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | Transformer 鏂囨湰鐢熸垚 | [toy_causal_lm_transformer](tracks/llm/lesson_01_toy_causal_lm_transformer/) | Causal Mask, 鑷洖褰掕В鐮?|

> [!NOTE]
> `resources/pdfs/llms/` 涓嬩繚鐣欎簡 50+ 绡?LLM 鐩稿叧璁烘枃涓庣瑪璁帮紝鍖呮嫭 PaLM銆佸ぇ妯″瀷缁艰堪绛夛紝鍙綔涓哄欢浼搁槄璇汇€?

---

### 馃寪 8. Multimodal / 澶氭ā鎬?

> 浠?CLIP 鍙屽瀵归綈鍒?LLaVA 鎸囦护璺熼殢锛屽啀鍒板紑鏀捐瘝姹囨娴嬨€佹椂搴忓畾浣?鈥?16 姝ヨ蛋瀹岀幇浠ｈ瑙夎瑷€寤烘ā鏍稿績鑴夌粶銆?

| 搴忓彿 | 椤圭洰 | 浠ｇ爜鏂囨。 | 鏍稿績姒傚康 |
|------|------|----------|----------|
| 1 | CLIP-Style Retrieval | [lesson_01_clip_toy_retrieval](tracks/multimodal/lesson_01_clip_toy_retrieval/) | 瀵规瘮瀛︿範, 鍙屽缂栫爜鍣?|
| 2 | BLIP-Lite Captioning + ITM | [lesson_02_blip_toy_captioning](tracks/multimodal/lesson_02_blip_toy_captioning/) | 瑙嗚 token 铻嶅悎, ITM |
| 3 | LLaVA-Lite Instruction VLM | [lesson_03_llava_toy_instruction_vlm](tracks/multimodal/lesson_03_llava_toy_instruction_vlm/) | 瑙嗚鍓嶇紑, 鎸囦护璺熼殢 |
| 4 | Grounding Referring | [lesson_04_grounding_toy_refexp](tracks/multimodal/lesson_04_grounding_toy_refexp/) | 鎸囦唬琛ㄨ揪, Box 鍥炲綊 |
| 5 | Mask Grounding | [lesson_05_mask_grounding_toy_refexp](tracks/multimodal/lesson_05_mask_grounding_toy_refexp/) | 鏂囨湰鏉′欢 Mask 棰勬祴 |
| 6 | Flamingo Interleaved VLM | [lesson_06_flamingo_toy_interleaved_vlm](tracks/multimodal/lesson_06_flamingo_toy_interleaved_vlm/) | 浜ら敊鍥炬枃, Few-shot |
| 7 | Q-Former Bridge VLM | [lesson_07_qformer_toy_bridge_vlm](tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/) | Cross-attention 鐡堕 |
| 8 | Perceiver Resampler VLM | [lesson_08_perceiver_resampler_toy_vlm](tracks/multimodal/lesson_08_perceiver_resampler_toy_vlm/) | 澶氳鍥?token 姹犲寲 |
| 9 | PaliGemma Multitask VLM | [lesson_09_paligemma_toy_siglip_decoder_vlm](tracks/multimodal/lesson_09_paligemma_toy_siglip_decoder_vlm/) | 鎻愮ず寮忓浠诲姟 |
| 10 | OWL-ViT Open-Vocab Detection | [lesson_10_owlvit_toy_open_vocab_detection](tracks/multimodal/lesson_10_owlvit_toy_open_vocab_detection/) | 寮€鏀捐瘝姹囨娴?|
| 11 | Grounded-SAM Segmentation | [lesson_11_grounded_sam_toy_open_vocab_segmentation](tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/) | 寮€鏀捐瘝姹囧垎鍓?|
| 12 | Key-Value OCR Document VLM | [lesson_12_key_value_ocr_toy_doc_vlm](tracks/multimodal/lesson_12_key_value_ocr_toy_doc_vlm/) | 鏂囨。瀛楁鎻愬彇 |
| 13 | Video VLM Temporal QA | [lesson_13_video_vlm_toy_temporal_qa](tracks/multimodal/lesson_13_video_vlm_toy_temporal_qa/) | 鐭棰?QA |
| 14 | BMN Temporal Grounding | [lesson_14_bmn_toy_temporal_grounding](tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/) | 鏃跺簭瀹氫綅, 杈圭晫棰勬祴 |
| 15 | 2D-TAN Temporal Grounding | [lesson_15_2dtan_toy_temporal_grounding](tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/) | 瀵嗛泦鏃跺簭娈靛浘 |
| 16 | Multi-Scale 2D-TAN | [lesson_16_multiscale_2dtan_toy_temporal_grounding](tracks/multimodal/lesson_16_multiscale_2dtan_toy_temporal_grounding/) | 澶氬昂搴︽椂搴忛噾瀛楀 |

```bash
# 鍐掔儫娴嬭瘯 Multimodal lesson
python -m tracks.multimodal.lesson_01_clip_toy_retrieval.train \
  --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1
```

<details>
<summary><b>VLM Zoo 鈥?70 涓瑙夎瑷€妯″瀷鏃忥紙鏁欏瀹炵幇 + 鏃堕棿绾匡級</b></summary>

| Family | 骞翠唤 | 鏍稿績鍒涙柊 |
|--------|------|---------|
| CLIP | 2021 | 瀵规瘮鍥炬枃棰勮缁?|
| ALIGN | 2021 | 澶ц妯″櫔澹板姣斿涔?|
| ViLT | 2021 | Patch 绾ц瑙夎瑷€ Transformer |
| SimVLM | 2021 | 绠€鍗曡瑙夎瑷€棰勮缁?|
| ALBEF | 2021 | 鍏堝榻愬啀铻嶅悎 |
| LiT | 2022 | 閿佸畾鍥惧儚鐨勬枃鏈井璋?|
| BLIP | 2022 | 寮曞寮忓浘鏂囬璁粌 |
| CoCa | 2022 | 瀵规瘮寮忔弿杩板櫒 |
| OFA | 2022 | 缁熶竴鏋舵瀯銆佷换鍔°€佹ā鎬?|
| Flamingo | 2022 | 浜ら敊鍥炬枃瑙嗚璇█妯″瀷 |
| PaLI | 2022 | Pathways 鍥炬枃妯″瀷 |
| BLIP-2 | 2023 | Q-Former 妗ユ帴瑙嗚涓?LLM |
| InstructBLIP | 2023 | 鎸囦护寰皟 BLIP-2 |
| LLaVA | 2023 | 瑙嗚鎸囦护寰皟 |
| MiniGPT-4 | 2023 | 鎶曞奖鍓嶇紑瑙嗚 LLM |
| Kosmos-2 | 2023 | 鎺ュ湴澶氭ā鎬?LLM |
| mPLUG-Owl2 | 2023 | 妯℃€佽嚜閫傚簲妯″潡 |
| CogVLM | 2023 | LLM 灞傚唴瑙嗚涓撳 |
| PaLI-X | 2023 | 缂╂斁鐗?Pathways 鍥炬枃妯″瀷 |
| Qwen-VL | 2023 | 閫氫箟鍗冮棶瑙嗚璇█妯″瀷 |
| Ferret | 2023 | 鎸囩偣寮忓尯鍩熸劅鐭ヨ瑙夎瑷€寤烘ā |
| Emu2 | 2023 | 澶氭ā鎬佺敓鎴愪笌鐞嗚В缁熶竴 |
| Fuyu | 2023 | 鍘熺敓 patch 搴忓垪瑙嗚杈撳叆 |
| IDEFICS2 | 2024 | 寮€鏀惧紡澶氬浘瀵硅瘽鍔╂墜 |
| InternVL | 2024 | 澶氬昂搴﹂珮鍒嗚鲸鐜囪瑙夌紪鐮?|
| Phi-3-Vision | 2024 | 杞婚噺瑙嗚璇█鎺ㄧ悊 |
| Janus | 2024 | 鐞嗚В涓庣敓鎴愮粺涓€瑙嗚鍓嶇 |
| Ovis | 2024 | 鏂囨。/OCR 鍦烘櫙浼樺寲鐨勮瑙夎瑷€鍔╂墜 |
| Cambrian | 2024 | 澶氳瑙夊铻嶅悎涓庤捀棣?|
| Molmo | 2024 | 寮€鏀炬暟鎹厤鏂归┍鍔ㄧ殑澶氭ā鎬佸姪鎵?|
| Video-LLaVA | 2024 | 瑙嗛鏃跺簭瑙嗚鎸囦护璺熼殢 |
| DeepSeek-VL | 2024 | 瀵硅瘽寮忓妯℃€佹帹鐞?|
| Qwen2-VL | 2024 | 鏇村己鏂囨。涓庤棰戠悊瑙?|
| VILA | 2024 | 杞婚噺瑙嗚璇█鍔╂墜 |
| Omni-VLM | 2024 | 缁熶竴澶氭ā鎬佺悊瑙ｆ帴鍙?|
| SEED-VL | 2024 | 寮哄寲妫€绱笌鐢熸垚缁熶竴 |
| MiniCPM-V | 2024 | 杞婚噺绔晶瑙嗚璇█妯″瀷 |
| Eagle-VLM | 2024 | Agent 椋庢牸澶氭ā鎬佸搷搴?|
| Phi-4-MM | 2025 | 杞婚噺澶氭ā鎬佹帹鐞嗗崌绾?|
| XComposer2 | 2025 | 缁嗙矑搴﹀浘鏂囩紪杈戜笌鐞嗚В |
| LLaVA-Next | 2025 | 鏇村己澶氬浘涓庤棰戠悊瑙?|
| IDEFICS3 | 2025 | 澶氬浘瀵硅瘽鏂颁竴浠ｆ帴鍙?|
| Kimi-VL | 2025 | 闀夸笂涓嬫枃澶氭ā鎬佸姪鎵?|
| Stem-VL | 2025 | 缁撴瀯鍖栧妯℃€佹帹鐞嗗師鍨?|
| Moondream2 | 2025 | 灏忓瀷绔晶瑙嗚闂瓟鍔╂墜 |
| Granite-Vision | 2025 | 浼佷笟鏂囨。涓庡浘琛ㄧ悊瑙?|
| OLMOCR | 2025 | 鏂囨。 OCR 涓撻」瑙嗚璇█妯″瀷 |
| InternLM-XComposer | 2025 | 澶氭ā鎬佸啓浣滀笌缂栬緫鍔╂墜 |
| MobileVLM | 2025 | 杞婚噺绉诲姩绔妯℃€佹ā鍨?|
| MiniCPM-O | 2025 | 绔晶寮€鏀惧紡澶氭ā鎬佹ā鍨?|
| Kosmos-2.5 | 2025 | 鏂囨。鐞嗚В涓?OCR 澧炲己 |
| ChartVLM | 2025 | 鍥捐〃鐞嗚В涓庢暟鎹棶绛?|
| DocOwl2 | 2025 | 鏂囨。闂瓟涓庣増闈㈢悊瑙?|
| Grounded-VLM | 2025 | 瀹氫綅澧炲己鐨勮瑙夎瑷€鎺ㄧ悊 |
| MetaVLM | 2025 | 鍏冨涔犲紡瑙嗚璇█閫傞厤 |
| Evo-VL | 2025 | 杩涘寲寮忓妯℃€佹帹鐞?|
| Agent-VL | 2025 | 闈㈠悜宸ュ叿璋冪敤鐨勫妯℃€佷唬鐞?|
| Video-Qwen-VL | 2025 | 瑙嗛澧炲己鐗堥€氫箟瑙嗚璇█妯″瀷 |
| SigLIP-VLM | 2025 | SigLIP 椋庢牸瀵归綈涓庣敓鎴愮粺涓€ |
| OCRVLM | 2025 | 鏂囨。 OCR 涓撻」澶氭ā鎬佸姪鎵?|
| Science-VLM | 2025 | 绉戝鍥捐〃涓庡疄楠屽浘鍍忕悊瑙?|
| WebVLM | 2025 | 缃戦〉鎴浘涓庣晫闈㈢悊瑙?|
| MixVLM | 2025 | 澶氳矾瑙嗚缂栫爜娣峰悎铻嶅悎 |
| EdgeVLM | 2025 | 绔晶杞婚噺澶氭ā鎬佹帹鐞?|
| InternVL2 | 2024 | 澶氬昂搴﹀妯℃€佸崌绾х増 |
| XGen-MM | 2024 | 鎸囦护璺熼殢澶氭ā鎬佹ā鍨?|
| Aria | 2024 | 绔埌绔瑙夊璇濆姪鎵?|
| LLaMA-Vision | 2024 | LLaMA 绯昏瑙夋墿灞?|
| Bunny | 2024 | 灏忓瀷瑙嗚鎸囦护妯″瀷 |
| Rabbit-VLM | 2025 | Agent 椋庢牸澶氭ā鎬佷氦浜?|

> 瀹屾暣鍒楄〃涓庡彉浣撹 `python scripts/vlm_zoo.py --list`

</details>

---

## Model Zoo

> 鍏ㄩ鍩熺粺涓€妯″瀷鍔ㄧ墿鍥?鈥?绾?PyTorch 鏈湴瀹炵幇锛屾棤闇€涓嬭浇棰勮缁冩潈閲嶏紝2500+ 鏋舵瀯 ID 涓€琛屽垏鎹?

### Zoo 瀛愮郴缁熸€昏锛?1 涓瓙绯荤粺锛?

| 棰嗗煙 | 瀛愮郴缁?| 绠楁硶鏃?| CLI 鑴氭湰 |
|------|--------|--------|---------|
| Vision | Backbones | 208 鏃?/ 736 IDs | `scripts/vision_zoo.py` |
| Vision | Detection (2D) | ~140 | `scripts/detection_zoo.py` |
| Vision | Instance Segmentation | 60 | `scripts/instance_segmentation_zoo.py` |
| Vision | Panoptic Segmentation | 60 | `scripts/panoptic_segmentation_zoo.py` |
| Vision | Lane Detection | 44 | `scripts/lane_detection_zoo.py` |
| Vision | Co-segmentation | 26 | `scripts/co_segmentation_zoo.py` |
| Vision | Fine-Grained Recognition | 112 | `scripts/fine_grained_recognition_zoo.py` |
| Vision | Action Recognition | 62 | `scripts/action_recognition_zoo.py` |
| Vision | MOT (2D) | 100 | `scripts/mot_zoo.py` |
| NLP | Text Encoders | 49 鏃?/ 813 IDs | `scripts/nlp_zoo.py` |
| Point Cloud | Backbones | 30 鏃?/ 64 IDs | `scripts/pointcloud_zoo.py` |
| Point Cloud | 3D Detection | 60 | `scripts/detection3d_zoo.py` |
| Point Cloud | 3D Segmentation | 60 | `scripts/segmentation3d_zoo.py` |
| Point Cloud | 3D Instance Seg | 50 | `scripts/instance_segmentation3d_zoo.py` |
| Point Cloud | 3D Tracking | 140 | `scripts/tracking3d_zoo.py` |
| Point Cloud | Gaussian Splatting | 10 | `dlhub/pointcloud/gaussian_splatting_zoo.py` |
| Multimodal | VLM | 70 | `scripts/vlm_zoo.py` |
| Multimodal | Prompt Learning | 10 | `dlhub/multimodal/prompt_learning_zoo.py` |
| Vision | New Directions Batch XIII | 80 | `dlhub/vision/*_zoo.py` |
| Generative | GAN | 44 | `scripts/gan_zoo.py` |
| Generative | Diffusion | 32 | `scripts/diffusion_zoo.py` |
| Federated | FL Strategies | 76 | `scripts/federated_zoo.py` |

鎵€鏈?Zoo 閬靛惊鐩稿悓鐨勮璁℃ā寮忥細

- **涓€鏂囦欢涓€绠楁硶鏃?* 鈥?濡?`resnet.py` 鍖呭惈 ResNet-18/34/50/101 鎵€鏈夊彉浣?
- **Lazy Import** 鈥?浠呭湪浣跨敤鏃跺姞杞斤紝鍚姩闆跺紑閿€
- **缁熶竴鎺ュ彛** 鈥?`build(arch_id, num_classes=...)` 鍗冲彲鏋勫缓浠绘剰妯″瀷
- **CLI 宸ュ叿** 鈥?`--list` 鍒楄〃銆乣--search` 鎼滅储銆乣--smoke` 鍐掔儫娴嬭瘯

#### Emerging Research Directions / 鏂扮爺绌舵柟鍚?

> 杩欎竴鎵硅ˉ鍏呯殑鏄鍓嶅皻鏈郴缁熷睍寮€鐨勬柟鍚戯紝姣忎釜鏂瑰悜鍏堣惤鍦?10 涓?toy-first family锛屼究浜庡悗缁户缁墿灞曘€?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| ReID / 琛屼汉閲嶈瘑鍒?| 10 | `dlhub/vision/reid/` |
| OCR / 鏂囧瓧璇嗗埆 | 10 | `dlhub/vision/ocr/` |
| Depth Estimation / 娣卞害浼拌 | 10 | `dlhub/vision/depth_estimation/` |
| Dehazing / 鍘婚浘 | 10 | `dlhub/vision/dehazing/` |
| Deblurring / 鍘绘ā绯?| 10 | `dlhub/vision/deblurring/` |
| Saliency Detection / 鏄捐憲鎬ф娴?| 10 | `dlhub/vision/saliency_detection/` |
| Anomaly Detection / 寮傚父妫€娴?| 10 | `dlhub/vision/anomaly_detection/` |
| Image Retrieval / 鍥惧儚妫€绱?| 10 | `dlhub/vision/image_retrieval/` |
| Medical Segmentation / 鍖诲鍒嗗壊 | 10 | `dlhub/vision/medical_segmentation/` |
| Remote Sensing Detection / 閬ユ劅妫€娴?| 10 | `dlhub/vision/remote_sensing_detection/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堜簩锛?
> 杩欎竴鎵圭户缁寜鈥滀竴涓?worktree 涓€涓柟鍚戔€濊ˉ鍏呭叏鏂版柟鍚戯紝姣忎釜鏂瑰悜鍚屾牱鍏堣惤鍦?10 涓?family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| HOI Detection / 浜虹墿浜や簰妫€娴?| 10 | `dlhub/vision/hoi_detection/` |
| Weakly Supervised Detection / 寮辩洃鐫ｆ娴?| 10 | `dlhub/vision/weakly_supervised_detection/` |
| Weakly Supervised Segmentation / 寮辩洃鐫ｅ垎鍓?| 10 | `dlhub/vision/weakly_supervised_segmentation/` |
| Video Object Segmentation / 瑙嗛鐩爣鍒嗗壊 | 10 | `dlhub/vision/video_object_segmentation/` |
| Crowd Counting / 浜虹兢璁℃暟 | 10 | `dlhub/vision/crowd_counting/` |
| Face Detection / 浜鸿劯妫€娴?| 10 | `dlhub/vision/face_detection/` |
| Face Alignment / 浜鸿劯瀵归綈 | 10 | `dlhub/vision/face_alignment/` |
| Human Pose Estimation / 浜轰綋濮挎€佷及璁?| 10 | `dlhub/vision/human_pose_estimation/` |
| Video Restoration / 瑙嗛淇 | 10 | `dlhub/vision/video_restoration/` |
| Geo-localization / 鍦扮悊瀹氫綅 | 10 | `dlhub/vision/geo_localization/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堜笁锛?
> 缁х画娌跨敤鈥滀竴鏂瑰悜涓€ worktree鈥濈殑鏂瑰紡琛ュ叏鏂颁换鍔″寘锛屾瘡涓柟鍚戝厛琛?10 涓?family 浣滀负绗竴鎵归鏋躲€?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Text Detection / 鏂囨湰妫€娴?| 10 | `dlhub/vision/text_detection/` |
| Text Recognition / 鏂囨湰璇嗗埆 | 10 | `dlhub/vision/text_recognition/` |
| Video Instance Segmentation / 瑙嗛瀹炰緥鍒嗗壊 | 10 | `dlhub/vision/video_instance_segmentation/` |
| 3D Pose Estimation / 3D 濮挎€佷及璁?| 10 | `dlhub/vision/pose_estimation_3d/` |
| 6D Pose Estimation / 6D 濮挎€佷及璁?| 10 | `dlhub/vision/sixd_pose_estimation/` |
| Face Anti-Spoofing / 娲讳綋妫€娴?| 10 | `dlhub/vision/face_anti_spoofing/` |
| Facial Expression Recognition / 琛ㄦ儏璇嗗埆 | 10 | `dlhub/vision/facial_expression_recognition/` |
| Person Attribute Recognition / 琛屼汉灞炴€ц瘑鍒?| 10 | `dlhub/vision/person_attribute_recognition/` |
| License Plate Recognition / 杞︾墝璇嗗埆 | 10 | `dlhub/vision/license_plate_recognition/` |
| Sketch Retrieval / 鑽夊浘妫€绱?| 10 | `dlhub/vision/sketch_retrieval/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝洓锛?
> 缁х画娌跨敤鈥滀竴鏂瑰悜涓€ worktree鈥濈殑鏂瑰紡鎵╁睍姝ゅ墠鏈缓鍖呯殑瑙嗚浠诲姟锛屾瘡涓柟鍚戝厛琛?10 涓?family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Image Matting / 鍥惧儚鎶犲浘 | 10 | `dlhub/vision/image_matting/` |
| Image Harmonization / 鍥惧儚鍗忚皟 | 10 | `dlhub/vision/image_harmonization/` |
| Image Inpainting / 鍥惧儚淇 | 10 | `dlhub/vision/image_inpainting/` |
| Image Fusion / 鍥惧儚铻嶅悎 | 10 | `dlhub/vision/image_fusion/` |
| Image Stitching / 鍥惧儚鎷兼帴 | 10 | `dlhub/vision/image_stitching/` |
| Temporal Action Localization / 鏃跺簭鍔ㄤ綔瀹氫綅 | 10 | `dlhub/vision/temporal_action_localization/` |
| Gaze Estimation / 瑙嗙嚎浼拌 | 10 | `dlhub/vision/gaze_estimation/` |
| Trajectory Prediction / 杞ㄨ抗棰勬祴 | 10 | `dlhub/vision/trajectory_prediction/` |
| Scene Graph Generation / 鍦烘櫙鍥剧敓鎴?| 10 | `dlhub/vision/scene_graph_generation/` |
| Camouflaged Object Detection / 浼鐗╀綋妫€娴?| 10 | `dlhub/vision/camouflaged_object_detection/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堜簲锛?
> 杩欎竴鎵圭户缁嫇灞曟鍓嶆湭寤哄寘鐨勬柟鍚戯紝瑕嗙洊缂栬緫銆佽瀺鍚堛€佸尮閰嶃€佸畾浣嶅拰鏃跺簭鐞嗚В绫讳换鍔°€?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Image Editing / 鍥惧儚缂栬緫 | 10 | `dlhub/vision/image_editing/` |
| Multi-focus Fusion / 澶氱劍鐐瑰浘鍍忚瀺鍚?| 10 | `dlhub/vision/multi_focus_fusion/` |
| Online Handwriting Recognition / 鑱旀満鎵嬪啓姹夊瓧璇嗗埆 | 10 | `dlhub/vision/online_handwriting_recognition/` |
| Lane Topology Estimation / 杞﹂亾鍥句及璁?| 10 | `dlhub/vision/lane_topology_estimation/` |
| Remote Sensing Change Detection / 閬ユ劅鍙樺寲妫€娴?| 10 | `dlhub/vision/remote_sensing_change_detection/` |
| Cross-view Geo-localization / 璺ㄨ鍥惧湴鐞嗗畾浣?| 10 | `dlhub/vision/cross_view_geo_localization/` |
| Video Understanding / 瑙嗛鐞嗚В | 10 | `dlhub/vision/video_understanding/` |
| Video Enhancement / 瑙嗛澧炲己 | 10 | `dlhub/vision/video_enhancement/` |
| Image Matching / 鍥惧儚鍖归厤 | 10 | `dlhub/vision/image_matching/` |
| Feature Matching / 鐗瑰緛鍖归厤 | 10 | `dlhub/vision/feature_matching/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝叚锛?
> 缁х画琛ュ叏姝ゅ墠鏈缓鍖呯殑鐢熸垚寮?鐞嗚В寮忚瑙変换鍔★紝姣忎釜鏂瑰悜浠嶇劧鍏堣惤鍦?10 涓?family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Low-light Enhancement / 浣庡厜澧炲己 | 10 | `dlhub/vision/low_light_enhancement/` |
| Image Colorization / 鍥惧儚涓婅壊 | 10 | `dlhub/vision/image_colorization/` |
| Referring Expression Comprehension / 鎸囦唬琛ㄨ揪鐞嗚В | 10 | `dlhub/vision/referring_expression_comprehension/` |
| Referring Expression Segmentation / 鎸囦唬琛ㄨ揪鍒嗗壊 | 10 | `dlhub/vision/referring_expression_segmentation/` |
| Open-vocabulary Segmentation / 寮€鏀捐瘝姹囧垎鍓?| 10 | `dlhub/vision/open_vocabulary_segmentation/` |
| Video Temporal Grounding / 瑙嗛鏃跺簭瀹氫綅 | 10 | `dlhub/vision/video_temporal_grounding/` |
| Document Understanding / 鏂囨。鐞嗚В | 10 | `dlhub/vision/document_understanding/` |
| Shadow Removal / 闃村奖鍘婚櫎 | 10 | `dlhub/vision/shadow_removal/` |
| Reflection Removal / 鍙嶅厜鍘婚櫎 | 10 | `dlhub/vision/reflection_removal/` |
| Novel View Synthesis / 鏂拌瑙掑悎鎴?| 10 | `dlhub/vision/novel_view_synthesis/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堜竷锛?
> 缁х画鍚戞鍓嶆湭寤哄寘鐨勭粏鍒嗚瑙夋柟鍚戞墿灞曪紝鑱氱劍鍖归厤銆佽В鏋愩€侀棶绛斿拰璺ㄦā鎬佸畾浣嶇被浠诲姟銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Optical Flow / 鍏夋祦浼拌 | 10 | `dlhub/vision/optical_flow/` |
| Person Search / 琛屼汉鎼滅储 | 10 | `dlhub/vision/person_search/` |
| Human Parsing / 浜轰綋瑙ｆ瀽 | 10 | `dlhub/vision/human_parsing/` |
| Scene Text Spotting / 鍦烘櫙鏂囨湰妫€娴嬭瘑鍒竴浣撳寲 | 10 | `dlhub/vision/scene_text_spotting/` |
| Stereo Matching / 鍙岀洰鍖归厤 | 10 | `dlhub/vision/stereo_matching/` |
| Video Captioning / 瑙嗛鎻忚堪 | 10 | `dlhub/vision/video_captioning/` |
| Video Question Answering / 瑙嗛闂瓟 | 10 | `dlhub/vision/video_question_answering/` |
| Few-shot Recognition / 灏忔牱鏈瘑鍒?| 10 | `dlhub/vision/few_shot_recognition/` |
| Interactive Segmentation / 浜や簰寮忓垎鍓?| 10 | `dlhub/vision/interactive_segmentation/` |
| Human Mesh Recovery / 浜轰綋缃戞牸鎭㈠ | 10 | `dlhub/vision/human_mesh_recovery/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝叓锛?
> 缁х画鎵╁睍姝ゅ墠鏈缓鍖呯殑鎰熺煡璐ㄩ噺銆佽法妯℃€佹帹鐞嗕笌鍑犱綍鐞嗚В浠诲姟锛屾瘡涓柟鍚戜粛鐒跺厛琛?10 涓?family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Image Quality Assessment / 鍥惧儚璐ㄩ噺璇勪及 | 10 | `dlhub/vision/image_quality_assessment/` |
| Aesthetic Assessment / 缇庡璇勫垎 | 10 | `dlhub/vision/aesthetic_assessment/` |
| Video Quality Assessment / 瑙嗛璐ㄩ噺璇勪及 | 10 | `dlhub/vision/video_quality_assessment/` |
| Visual Dialog / 瑙嗚瀵硅瘽 | 10 | `dlhub/vision/visual_dialog/` |
| Visual Entailment / 瑙嗚钑村惈 | 10 | `dlhub/vision/visual_entailment/` |
| Image Captioning / 鍥惧儚鎻忚堪 | 10 | `dlhub/vision/image_captioning/` |
| Phrase Grounding / 鐭瀹氫綅 | 10 | `dlhub/vision/phrase_grounding/` |
| Depth Completion / 娣卞害琛ュ叏 | 10 | `dlhub/vision/depth_completion/` |
| Surface Normal Estimation / 娉曠嚎浼拌 | 10 | `dlhub/vision/surface_normal_estimation/` |
| Point Cloud Registration / 鐐逛簯閰嶅噯 | 10 | `dlhub/pointcloud/registration/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堜節锛?
> 缁х画瑕嗙洊姝ゅ墠鏈缓鍖呯殑璐ㄩ噺璇勪及銆佽瑙夋帹鐞嗕笌琛ュ叏绫讳换鍔★紝姣忎釜鏂瑰悜渚濇棫鍏堣ˉ 10 涓?family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Image Quality Assessment / 鍥惧儚璐ㄩ噺璇勪及 | 10 | `dlhub/vision/image_quality_assessment/` |
| Aesthetic Assessment / 缇庡璇勫垎 | 10 | `dlhub/vision/aesthetic_assessment/` |
| Video Quality Assessment / 瑙嗛璐ㄩ噺璇勪及 | 10 | `dlhub/vision/video_quality_assessment/` |
| Visual Dialog / 瑙嗚瀵硅瘽 | 10 | `dlhub/vision/visual_dialog/` |
| Visual Entailment / 瑙嗚钑村惈 | 10 | `dlhub/vision/visual_entailment/` |
| Image Captioning / 鍥惧儚鎻忚堪 | 10 | `dlhub/vision/image_captioning/` |
| Phrase Grounding / 鐭瀹氫綅 | 10 | `dlhub/vision/phrase_grounding/` |
| Depth Completion / 娣卞害琛ュ叏 | 10 | `dlhub/vision/depth_completion/` |
| Surface Normal Estimation / 娉曠嚎浼拌 | 10 | `dlhub/vision/surface_normal_estimation/` |
| Point Cloud Registration / 鐐逛簯閰嶅噯 | 10 | `dlhub/pointcloud/registration/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝崄锛?
> 缁х画鎵╁睍姝ゅ墠鏈缓鍖呯殑搴曞眰缁撴瀯鎰熺煡涓庝笓涓氳瑙変换鍔★紝姣忎釜鏂瑰悜浠嶇劧鍏堣ˉ 10 涓?family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Edge Detection / 杈圭紭妫€娴?| 10 | `dlhub/vision/edge_detection/` |
| Line Segment Detection / 绾挎妫€娴?| 10 | `dlhub/vision/line_segment_detection/` |
| Contour Detection / 杞粨妫€娴?| 10 | `dlhub/vision/contour_detection/` |
| Defect Detection / 缂洪櫡妫€娴?| 10 | `dlhub/vision/defect_detection/` |
| Document Layout Analysis / 鏂囨。鐗堥潰鍒嗘瀽 | 10 | `dlhub/vision/document_layout_analysis/` |
| Table Structure Recognition / 琛ㄦ牸缁撴瀯璇嗗埆 | 10 | `dlhub/vision/table_structure_recognition/` |
| Chart Understanding / 鍥捐〃鐞嗚В | 10 | `dlhub/vision/chart_understanding/` |
| Fashion Compatibility / 鏃跺皻鎼厤棰勬祴 | 10 | `dlhub/vision/fashion_compatibility/` |
| Food Recognition / 椋熺墿璇嗗埆 | 10 | `dlhub/vision/food_recognition/` |
| Symbol Recognition / 绗﹀彿璇嗗埆 | 10 | `dlhub/vision/symbol_recognition/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝崄涓€锛?
> 缁х画娌跨潃鈥滄鍓嶆湭寤哄寘鈥濈殑鏂瑰悜鎵╁睍锛岃ˉ鍏呯粨鏋勬劅鐭ャ€佷笓涓氳瘑鍒拰宸ヤ笟/鏂囨。瑙嗚浠诲姟銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Edge Detection / 杈圭紭妫€娴?| 10 | `dlhub/vision/edge_detection/` |
| Line Segment Detection / 绾挎妫€娴?| 10 | `dlhub/vision/line_segment_detection/` |
| Contour Detection / 杞粨妫€娴?| 10 | `dlhub/vision/contour_detection/` |
| Defect Detection / 缂洪櫡妫€娴?| 10 | `dlhub/vision/defect_detection/` |
| Document Layout Analysis / 鏂囨。鐗堥潰鍒嗘瀽 | 10 | `dlhub/vision/document_layout_analysis/` |
| Table Structure Recognition / 琛ㄦ牸缁撴瀯璇嗗埆 | 10 | `dlhub/vision/table_structure_recognition/` |
| Chart Understanding / 鍥捐〃鐞嗚В | 10 | `dlhub/vision/chart_understanding/` |
| Fashion Compatibility / 鏃跺皻鎼厤棰勬祴 | 10 | `dlhub/vision/fashion_compatibility/` |
| Food Recognition / 椋熺墿璇嗗埆 | 10 | `dlhub/vision/food_recognition/` |
| Symbol Recognition / 绗﹀彿璇嗗埆 | 10 | `dlhub/vision/symbol_recognition/` |

#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝崄浜岋級
> 缁х画娌跨潃鈥滄鍓嶆湭寤哄寘鈥濈殑瑙嗚-鏃剁┖-澶氭ā鎬佽竟鐣岋紝琛ュ叏鎻愮ず寤烘ā銆佸湴鐐硅瘑鍒€佽祫浜т笌鏂囨。鐞嗚В绛夋柊鏂瑰悜锛屾瘡涓柟鍚戜粛鐒跺厛琛?10 涓?family銆?
| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Visual Prompting / 瑙嗚鎻愮ず寤烘ā | 10 | `dlhub/vision/visual_prompting/` |
| Visual Place Recognition / 瑙嗚鍦扮偣璇嗗埆 | 10 | `dlhub/vision/visual_place_recognition/` |
| Video Prediction / 瑙嗛棰勬祴 | 10 | `dlhub/vision/video_prediction/` |
| Pose Tracking / 濮挎€佽窡韪?| 10 | `dlhub/vision/pose_tracking/` |
| Pedestrian Attribute Analysis / 琛屼汉灞炴€у垎鏋?| 10 | `dlhub/vision/pedestrian_attribute_analysis/` |
| Object Counting / 鐩爣璁℃暟 | 10 | `dlhub/vision/object_counting/` |
| Multimodal Fusion / 澶氭ā鎬佽瀺鍚?| 10 | `dlhub/vision/multimodal_fusion/` |
| Image Forensics / 鍥惧儚鍙栬瘉 | 10 | `dlhub/vision/image_forensics/` |
| Graphical Document Parsing / 鍥惧舰鏂囨。瑙ｆ瀽 | 10 | `dlhub/vision/graphical_document_parsing/` |
| Fine-Grained Retrieval / 缁嗙矑搴︽绱?| 10 | `dlhub/vision/fine_grained_retrieval/` |
#### Additional New Directions / 鏂板鐮旂┒鏂瑰悜锛堝崄涓夛級
> 杩欎竴鎵圭户缁粠鐢ㄦ埛缁欏嚭鐨勪富棰樻睜閲岃ˉ榻愪粨搴撳皻鏈缓鍖呯殑鏂瑰悜锛岃鐩栬棰戞彃甯с€佽棰戠ǔ鍍忋€佽棰戠洰鏍囨娴嬨€佹枃妗ｅ嚑浣曘€丄IGC銆丳rompt Learning 鍜?3DGS锛屾瘡涓柟鍚戝厛琛?10 涓?toy-first family銆?

| 鏂瑰悜 | 褰撳墠瀹舵棌鏁?| 鍖呰矾寰?|
|------|-----------|--------|
| Video Frame Interpolation / 瑙嗛鎻掑抚 | 10 | `dlhub/vision/video_frame_interpolation/` |
| Video Stabilization / 瑙嗛绋冲儚 | 10 | `dlhub/vision/video_stabilization/` |
| Video Object Detection / 瑙嗛鐩爣妫€娴?| 10 | `dlhub/vision/video_object_detection/` |
| Document Dewarping / 鏂囨。鐭 | 10 | `dlhub/vision/document_dewarping/` |
| Layout Generation / 甯冨眬鐢熸垚 | 10 | `dlhub/vision/layout_generation/` |
| Adversarial Robustness / 瀵规姉椴佹鎬?| 10 | `dlhub/vision/adversarial_robustness/` |
| Data Augmentation / 鏁版嵁澧炲箍 | 10 | `dlhub/vision/data_augmentation/` |
| Image Synthesis / 鍥惧儚鍚堟垚 | 10 | `dlhub/vision/image_synthesis/` |
| Prompt Learning / 澶氭ā鎬?Prompt Learning | 10 | `dlhub/multimodal/prompt_learning/` |
| Gaussian Splatting / 3DGS | 10 | `dlhub/pointcloud/gaussian_splatting/` |


---

### Vision Zoo / 736 Architectures

```bash
# 鍒楀嚭鎵€鏈夊彲鐢ㄦ灦鏋?
python scripts/vision_zoo.py --list

# 鎼滅储鐗瑰畾鏋舵瀯
python scripts/vision_zoo.py --search convnext

# 鍐掔儫娴嬭瘯
python scripts/vision_zoo.py --smoke resnet50
```

#### Fine-Grained Recognition (FGVC) Local Zoo

> 缁嗙矑搴﹁瑙夎瘑鍒紙FGVC锛夋ā鍨嬫棌琛ュ厖锛欱ilinear / Part-based / Transformer / Prompt / CLIP / MLLM reasoning锛坱oy-first, no downloads锛?

```bash
python scripts/fine_grained_recognition_zoo.py --list
python scripts/fine_grained_recognition_zoo.py --search transfg
python scripts/fine_grained_recognition_zoo.py --smoke dlfgvc:fine_r1_tiny
```

> 鏃堕棿绾夸笌鏂规硶璇存槑瑙?`dlhub/vision/fine_grained_recognition/README.md`

#### Action Recognition (Video + Skeleton) Local Zoo

> 琛屼负璇嗗埆锛堝姩浣滆瘑鍒級妯″瀷鏃忚ˉ鍏咃細Video (NCTHW) + Skeleton (NCTV)锛宼oy-first, no downloads

```bash
python scripts/action_recognition_zoo.py --list
python scripts/action_recognition_zoo.py --search stgcn
python scripts/action_recognition_zoo.py --smoke dlactv:c3d_tiny
python scripts/action_recognition_zoo.py --smoke dlacts:stgcn_tiny
```

> 鏃堕棿绾夸笌鏂规硶璇存槑瑙?`dlhub/vision/action_recognition/README.md`

#### Multi-Object Tracking (MOT) Local Zoo

> 澶氱洰鏍囪窡韪ā鍨嬫棌琛ュ厖锛?D 鍗曠浉鏈?MOT锛?00 绠楁硶鏃忥紙姣忔棌 `tiny/small/base`锛夛紝toy-first, no downloads

```bash
python scripts/mot_zoo.py --list
python scripts/mot_zoo.py --search bytetrack
python scripts/mot_zoo.py --timeline
python scripts/mot_zoo.py --recommend realtime --top-k 8 --variant tiny
python scripts/mot_zoo.py --recommend occlusion --top-k 8 --variant tiny --emit-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --skip-existing
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --summary-only
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --rank-by loss
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-leaderboard outputs/vision/mot_leaderboard.json
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir outputs/vision/mot_artifacts
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir auto
python scripts/mot_zoo.py --smoke mot2d:sort_tiny
```

> 缁勫埆銆侀€夊瀷寤鸿涓?80 鏃忓垪琛ㄨ `dlhub/vision/mot/README.md`

#### Detection Zoo (2D)

> 2D 鐩爣妫€娴嬫ā鍨嬫棌锛欰nchor-based / Anchor-free / Transformer-based / 杞婚噺绾ф娴嬪櫒锛寏140 绠楁硶

```bash
python scripts/detection_zoo.py --list
python scripts/detection_zoo.py --search fcos
python scripts/detection_zoo.py --smoke dldet:fcos_r50_tiny
```

#### Instance & Panoptic Segmentation Zoo

> 瀹炰緥鍒嗗壊 + 鍏ㄦ櫙鍒嗗壊锛歁ask R-CNN / YOLACT / Panoptic FPN 绛?

```bash
# 瀹炰緥鍒嗗壊
python scripts/instance_segmentation_zoo.py --list
python scripts/instance_segmentation_zoo.py --smoke dlinsseg:maskrcnn_r50_tiny

# 鍏ㄦ櫙鍒嗗壊
python scripts/panoptic_segmentation_zoo.py --list
python scripts/panoptic_segmentation_zoo.py --smoke dlpanseg:panfpn_r50_tiny
```

#### Lane Detection Zoo

> 杞﹂亾绾挎娴嬫ā鍨嬫棌锛?4 绠楁硶鏃忥紝Anchor / Parametric / Segmentation / Keypoint / Transformer 浜斿ぇ鑼冨紡

```bash
python scripts/lane_detection_zoo.py --list
python scripts/lane_detection_zoo.py --search laneatt
python scripts/lane_detection_zoo.py --smoke dllane:laneatt_r18_tiny
```

#### Co-segmentation Zoo

> 鍗忓悓鍒嗗壊妯″瀷鏃忥細26 绠楁硶鏃忥紝Group / Pair 绾у埆鍥惧儚鍏卞垎鍓?

```bash
python scripts/co_segmentation_zoo.py --list
python scripts/co_segmentation_zoo.py --smoke dlcoseg:coatt_tiny
```

<details>
<summary><b>涓昏鏋舵瀯鍒嗙被</b></summary>

| 绫诲埆 | 浠ｈ〃鏋舵瀯 | 鏁伴噺 |
|------|---------|------|
| 缁忓吀 CNN | AlexNet, VGG, GoogLeNet, ResNet, DenseNet | ~60 |
| 楂樻晥缃戠粶 | MobileNet v1-v4, EfficientNet v1/v2, GhostNet, ShuffleNet | ~80 |
| 娉ㄦ剰鍔?CNN | SENet, CBAM, BAM, ECA-Net, SK-Net, CoordAtt | ~50 |
| 鐜颁唬 CNN | ConvNeXt v1/v2, RepVGG, RepLKNet, HorNet, FocalNet | ~40 |
| Vision Transformer | ViT, DeiT, BEiT, Swin v2, CSwin, CaiT, CrossViT | ~120 |
| 楂樻晥 Transformer | EfficientViT, TinyViT, EdgeViT, FastViT, SwiftFormer | ~60 |
| MLP 绯诲垪 | MLP-Mixer, gMLP, ResMLP, FNet, CycleMLP, WaveMLP | ~50 |
| Hybrid | CoAtNet, MobileFormer, Uniformer, MaxViT, MobileViT | ~60 |
| 鐗规畩缁撴瀯 | CapsNet, FractalNet, HRNet, NAS 绯诲垪, Mamba | ~50 |

</details>

---

### NLP Zoo / 813 Architectures

```bash
# 鍒楀嚭鎵€鏈夊彲鐢ㄦ灦鏋?
python scripts/nlp_zoo.py --list

# 鎼滅储鐗瑰畾鏋舵瀯
python scripts/nlp_zoo.py --search bert

# 鍐掔儫娴嬭瘯
python scripts/nlp_zoo.py --smoke bert_base
```

<details>
<summary><b>涓昏鏋舵瀯鍒嗙被</b></summary>

| 绫诲埆 | 浠ｈ〃鏋舵瀯 |
|------|---------|
| Transformer | BERT, GPT, T5, ALBERT, DistilBERT, Longformer, BigBird |
| 楂樻晥 Transformer | Performer, Nystromformer, FNet, Synthesizer, Linformer |
| RNN 绯诲垪 | LSTM, GRU, BiLSTM, BiGRU, IndRNN, SRU, QRNN |
| CNN 绯诲垪 | TextCNN, InceptionCNN, DPCNN, VDCNN, ResConv |
| MLP 绯诲垪 | gMLP, ResMLP, MLP-Mixer |
| 杞婚噺绾?| FastText, WaveNet, TCN |

</details>

---

### Point Cloud Zoo / 64 Architectures

```bash
# 鍦?lesson_04 涓垏鎹?backbone
python -m tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train \
  --arch pointnet --dataset fake --epochs 1
```

> 璇︾粏鍒楄〃瑙?[Point Cloud Track](#-point-cloud--鐐逛簯) 鐨?Backbone 琛ㄦ牸銆?

#### 3D Detection Zoo

> 3D 鐩爣妫€娴嬫ā鍨嬫棌锛?0 绠楁硶鏃忥紝Point-based / Voxel-based / Pillar-based / Multi-modal

```bash
python scripts/detection3d_zoo.py --list
python scripts/detection3d_zoo.py --search pointpillars
python scripts/detection3d_zoo.py --smoke dldet3d:pointpillars_tiny
```

#### 3D Segmentation Zoo

> 3D 璇箟鍒嗗壊妯″瀷鏃忥細60 绠楁硶鏃忥紝Point / Voxel / Range-view / Fusion

```bash
python scripts/segmentation3d_zoo.py --list
python scripts/segmentation3d_zoo.py --search randlanet
python scripts/segmentation3d_zoo.py --smoke dlseg3d:randlanet_tiny
```

#### 3D Instance Segmentation Zoo

> 3D 瀹炰緥鍒嗗壊妯″瀷鏃忥細40 绠楁硶鏃忥紝Proposal-based / Grouping-based / Panoptic

```bash
python scripts/instance_segmentation3d_zoo.py --list
python scripts/instance_segmentation3d_zoo.py --smoke dlinsseg3d:pointgroup_tiny
```

#### 3D Tracking Zoo

> 3D 澶氱洰鏍囪窡韪ā鍨嬫棌锛?31 绠楁硶鏃忥紝LiDAR / Camera-LiDAR / Radar-LiDAR

```bash
python scripts/tracking3d_zoo.py --list
python scripts/tracking3d_zoo.py --search centerpoint
python scripts/tracking3d_zoo.py --smoke dltrk3d:centerpoint_tiny
```

---

### VLM Zoo / 70 Families

> 瑙嗚璇█妯″瀷鏃忥細70 涓?Family锛屼粠 CLIP 鍒?EdgeVLM锛岀函 PyTorch 鏁欏瀹炵幇

```bash
python scripts/vlm_zoo.py --list
python scripts/vlm_zoo.py --search llava
python scripts/vlm_zoo.py --timeline
python scripts/vlm_zoo.py --smoke dlvlm:clip_tiny
```

> 璇︾粏 Family 鍒楄〃瑙?[Multimodal Track](#-multimodal--澶氭ā鎬? 鐨?VLM Zoo 琛ㄦ牸銆?

---

### Generative Zoo / GAN + Diffusion

> 鐢熸垚妯″瀷鏃忥細GAN锛?4 绠楁硶鏃忥級+ Diffusion锛?2 绠楁硶鏃忥級锛岀函 PyTorch toy 瀹炵幇

```bash
# GAN Zoo
python scripts/gan_zoo.py --list
python scripts/gan_zoo.py --search stylegan
python scripts/gan_zoo.py --smoke dlgan:dcgan_tiny

# Diffusion Zoo
python scripts/diffusion_zoo.py --list
python scripts/diffusion_zoo.py --search ddpm
python scripts/diffusion_zoo.py --smoke dldiff:ddpm_tiny
```

<details>
<summary><b>GAN 涓昏鏋舵瀯</b></summary>

| 绫诲埆 | 浠ｈ〃鏋舵瀯 |
|------|---------|
| 鏃犳潯浠?GAN | DCGAN, WGAN, WGAN-GP, LSGAN, SNGAN |
| 鏉′欢 GAN | cGAN, ACGAN, InfoGAN, Pix2Pix |
| 鍥惧儚缈昏瘧 | CycleGAN, StarGAN, UNIT, MUNIT |
| 楂樺垎杈ㄧ巼 | ProGAN, StyleGAN, StyleGAN2, StyleGAN3 |
| 杞婚噺绾?| LightGAN, FastGAN |

</details>

<details>
<summary><b>Diffusion 涓昏鏋舵瀯</b></summary>

| 绫诲埆 | 浠ｈ〃鏋舵瀯 |
|------|---------|
| 鍩虹鎵╂暎 | DDPM, DDIM, Score-SDE |
| 鏉′欢鎵╂暎 | Classifier-Guided, Classifier-Free |
| 闅愮┖闂存墿鏁?| Latent Diffusion, Stable Diffusion |
| 蹇€熼噰鏍?| DPM-Solver, Consistency Models |

</details>

---

## Federated Learning Zoo

> 鑱旈偊瀛︿範绛栫暐搴?鈥?76 绉嶈仈閭︿紭鍖?/ 涓€у寲 / 闅愮绛栫暐锛岀函 PyTorch 鏁欏瀹炵幇

```bash
python scripts/federated_zoo.py --list
python scripts/federated_zoo.py --search fedavg
python scripts/federated_zoo.py --timeline
```

<details>
<summary><b>鍏ㄩ儴 76 绉嶇瓥鐣ワ紙鎸?13 涓垎缁勶級</b></summary>

| 鍒嗙粍 | 绛栫暐 | 璇存槑 |
|------|------|------|
| **Optimization** | FedAvg | 杩唬寮忔ā鍨嬪钩鍧?|
| | FedProx | 杩戠姝ｅ垯鍖?FedAvg |
| | FedNova | 褰掍竴鍖栧钩鍧?|
| | FedDyn | 鍔ㄦ€佹鍒欏寲鑱旈偊浼樺寲 |
| **Server Optimizer** | FedAdam | 鏈嶅姟绔?Adam |
| | FedYogi | 鏈嶅姟绔?Yogi |
| **Control Variate** | SCAFFOLD | 鎺у埗鍙橀噺淇瀹㈡埛绔紓绉?|
| **Feature Normalization** | FedBN | 鏈湴 Batch Normalization |
| **Personalization** | FedPer | Base/Head 鍒嗗壊涓€у寲 |
| | APFL | 鑷€傚簲涓€у寲鑱旈偊瀛︿範 |
| | Ditto | 杩戠鏈湴澶翠釜鎬у寲 |
| | pFedMe | 鍏冩鍒欏寲涓€у寲 |
| | MOON | 妯″瀷瀵规瘮涓€у寲 |
| | Per-FedAvg | 鍏冨涔犱釜鎬у寲 |
| | FedRep | 鍏变韩琛ㄧず + 涓€у寲澶?|
| | FedAMP | 娉ㄦ剰鍔涙秷鎭紶閫掍釜鎬у寲 |
| | FedProto | 鍘熷瀷鍖栬仈閭﹀涔?|
| | IFCA | 鑱氱被涓€у寲鑱旈偊瀛︿範 |
| **Fairness** | q-FedAvg | 鍏钩璧勬簮鍒嗛厤 |
| | AFL | 涓嶅彲鐭ヨ仈閭﹀涔?|
| | TERM | 鍊炬枩缁忛獙椋庨櫓鏈€灏忓寲 |
| **Long-tail Robustness** | FedRS | 绫讳笉骞宠　閲嶅钩琛?Softmax |
| | FedLC | 闀垮熬 Logit 鏍″噯 |
| | FedRoD | 椴佹钂搁 |
| **Split Learning** | SplitFed | 鑱旈偊鍒嗗壊瀛︿範 |
| | SplitFedV2 | 澧炲己鍒嗗壊鑱旈偊娣峰悎璁粌 |
| **Heterogeneous Width** | HeteroFL | 寮傛瀯瀹藉害鑱旈偊瀛︿範 |
| | FjORD | 鑱旈偊 Dropout |
| **Distillation** | FedGKT | 鑱旈偊缁勭煡璇嗚浆绉?|
| | FedDF | 闆嗘垚钂搁鑱旈偊瀛︿範 |
| **Privacy** | DP-FedAvg | 宸垎闅愮鑱旈偊骞冲潎 |
| | DP-FedProx | 宸垎闅愮杩戠鑱旈偊瀛︿範 |
| **Compression** | FedPAQ | 鍛ㄦ湡骞冲潎 + 閲忓寲 |
| | STC | 绋€鐤忎笁鍊煎帇缂?|
| **Secure Aggregation** | SecureAgg | 闅愮淇濇姢瀹夊叏姹傚拰 |
| | LightSecAgg | 杞婚噺瀹夊叏鑱氬悎 |

</details>

---

## NumPy ML Algorithms

> 绾?NumPy 鎵嬪啓缁忓吀鏈哄櫒瀛︿範绠楁硶 鈥?闆舵繁搴﹀涔犱緷璧栵紝鐞嗚В绠楁硶鏈川

| 绫诲埆 | 绠楁硶 | 鏂囦欢 | 鏍稿績鍘熺悊 |
|------|------|------|---------|
| **绾挎€фā鍨?* | Linear Regression | `linear_models.py` | 鏈€灏忎簩涔? 姊害涓嬮檷 |
| **绾挎€фā鍨?* | Ridge Regression | `linear_models.py` | L2 姝ｅ垯鍖? 闂紡瑙?|
| **绾挎€фā鍨?* | Logistic Regression | `linear_models.py` | Sigmoid, 浜ゅ弶鐔?|
| **绾挎€фā鍨?* | Softmax Regression | `linear_models.py` | Softmax, 澶氬垎绫讳氦鍙夌喌 |
| **鏍告柟娉?* | Linear SVM | `svm.py` | Hinge Loss, 鏈€澶ч棿闅?|
| **鏍戞ā鍨?* | Decision Tree | `decision_tree.py` | Gini 涓嶇函搴? 閫掑綊鍒嗚 |
| **闆嗘垚鏂规硶** | Random Forest | `random_forest.py` | Bagging, 鐗瑰緛闅忔満閲囨牱 |
| **闆嗘垚鏂规硶** | AdaBoost (Classification) | `adaboost.py` | Boosting, Decision Stumps |
| **闆嗘垚鏂规硶** | Gradient Boosting (Regression) | `gradient_boosting.py` | Boosting, 娈嬪樊鎷熷悎 |
| **姒傜巼妯″瀷** | Naive Bayes | `naive_bayes.py` | 鏉′欢鐙珛, 骞虫粦 |
| **姒傜巼妯″瀷** | GMM | `gmm.py` | EM 绠楁硶, 楂樻柉娣峰悎 |
| **鐢熸垚妯″瀷** | LDA / QDA | `discriminant_analysis.py` | 楂樻柉鍋囪, 鍒ゅ埆鍑芥暟 |
| **杩戦偦** | KNN | `knn.py` | 璺濈搴﹂噺, 澶氭暟鎶曠エ |
| **鑱氱被** | K-Means | `kmeans.py` | 璐ㄥ績杩唬, Lloyd 绠楁硶 |
| **鑱氱被** | K-Medoids | `kmedoids.py` | Medoid, PAM |
| **鑱氱被** | Agglomerative Clustering | `clustering.py` | 灞傛鑱氱被, Linkage |
| **鑱氱被** | DBSCAN | `clustering.py` | 瀵嗗害鑱氱被, 閭诲煙鎵╁睍 |
| **鑱氱被** | Spectral Clustering | `spectral_clustering.py` | 鍥炬媺鏅媺鏂? 鐗瑰緛鍚戦噺 |
| **闄嶇淮** | PCA | `pca.py` | 鐗瑰緛鍊煎垎瑙? 鏂瑰樊鏈€澶у寲 |
| **闄嶇淮** | NMF | `nmf.py` | 闈炶礋鍒嗚В, 涔樻硶鏇存柊 |
| **闄嶇淮** | FastICA | `ica.py` | 鐙珛鎴愬垎, Fixed-point |
| **闄嶇淮** | Isomap | `isomap.py` | 娴嬪湴璺濈, MDS |
| **搴忓垪妯″瀷** | Markov Chain | `markov_chain.py` | 杞Щ鐭╅樀, 骞虫粦 |
| **搴忓垪妯″瀷** | N-gram LM | `ngram.py` | 璁℃暟, Laplace 骞虫粦 |
| **搴忓垪妯″瀷** | Categorical HMM | `hmm.py` | Forward / Viterbi, log-space |
| **绁炵粡缃戠粶** | Perceptron | `perceptron.py` | 鎰熺煡鏈哄涔犺鍒?|
| **绁炵粡缃戠粶** | MLP | `mlp.py` | 鍙嶅悜浼犳挱, 閾惧紡娉曞垯 |

<sub>鎵€鏈夋枃浠朵綅浜?`ml_algorithms/python/`锛屼娇鐢?`@dataclass` 妯″紡瀹炵幇銆?/sub>

---

## Optimization Toolkit

> 绾?NumPy 瀹炵幇 鈥?鐞嗚В浼樺寲鍣ㄥ拰璋冨害鍣ㄧ殑鏁板鏈川

<table>
<tr>
<td valign="top" width="25%">

**Optimizers**
| 绠楁硶 | 鐗圭偣 |
|------|------|
| SGD | 鍩虹闅忔満姊害涓嬮檷 |
| Momentum | 鍔ㄩ噺鍔犻€?|
| RMSProp | 鑷€傚簲瀛︿範鐜?|
| Adagrad | 绋€鐤忔搴﹀弸濂?|
| Adam | Momentum + RMSProp |

</td>
<td valign="top" width="25%">

**LR Schedulers**
| 绛栫暐 | 鐗圭偣 |
|------|------|
| StepDecay | 闃舵寮忚“鍑?|
| ExponentialDecay | 鎸囨暟琛板噺 |
| CosineAnnealing | 浣欏鸡閫€鐏?|
| WarmupCosine | 棰勭儹 + 浣欏鸡 |

</td>
<td valign="top" width="25%">

**Losses**
| 鍑芥暟 | 鐢ㄩ€?|
|------|------|
| MSE | 鍥炲綊 |
| MAE | 椴佹鍥炲綊 |
| Binary CE | 浜屽垎绫?|
| Categorical CE | 澶氬垎绫?|

</td>
<td valign="top" width="25%">

**Metrics**
| 鎸囨爣 | 鐢ㄩ€?|
|------|------|
| Accuracy | 鍒嗙被鍑嗙‘鐜?|
| Precision | 绮剧‘鐜?|
| Recall / F1 | 鍙洖鐜?/ F1 |
| R虏 Score | 鍥炲綊鎷熷悎搴?|

</td>
</tr>
</table>

<details>
<summary><b>鏇村浼樺寲绠楁硶</b></summary>

| 绠楁硶 | 鐩綍 | 璇存槑 |
|------|------|------|
| 铓佺兢浼樺寲 (ACO) | `optimization/ACO/` | 鏃呰鍟嗛棶棰樻眰瑙ｏ紝鍚師鐞嗗浘 |
| 閬椾紶绠楁硶 (GA) | `optimization/GA/` | 杩涘寲鎼滅储锛屽惈娴佺▼鍥?|
| 绮掑瓙缇や紭鍖?(PSO) | `optimization/PSO/` | 缇や綋鏅鸿兘浼樺寲 |
| 灞傛鍒嗘瀽娉?(AHP) | `optimization/AHP/` | 澶氬噯鍒欏喅绛?|
| Lasso 浼樺寲 | `optimization/Lasso/` | L1 姝ｅ垯鍖栬矾寰勶紝鍚彲瑙嗗寲 |

</details>

---

## Documentation

| 鏂囨。 | 璇存槑 | 閫傚悎璋?|
|------|------|--------|
| [`ROADMAP.md`](docs/ROADMAP.md) | 瀛︿範璺嚎鍥句笌鎺ㄨ崘椤哄簭 | 鍒濆鑰?|
| [`INSTALL.md`](docs/INSTALL.md) | 瀹夎鎸囧崡 | 鎵€鏈変汉 |
| [`RUNNING.md`](docs/RUNNING.md) | 濡備綍杩愯 Lesson | 鎵€鏈変汉 |
| [`STRUCTURE.md`](docs/STRUCTURE.md) | 浠撳簱缁撴瀯璇﹁В | 鎯虫繁鍏ヤ簡瑙ｇ殑浜?|
| [`CONVENTIONS.md`](docs/CONVENTIONS.md) | 杩愯 & 瀹為獙绾﹀畾 | 璐＄尞鑰?|
| [`STYLEGUIDE.md`](docs/STYLEGUIDE.md) | 浠ｇ爜瑙勮寖 | 璐＄尞鑰?|
| [`faq.md`](docs/faq.md) | 甯歌闂 | 閬囧埌闂鏃?|

---

## Design Philosophy

```
              鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
              鈹?                  DL-Hub 璁捐鐞嗗康                      鈹?
              鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
              鈹?Offline-first 鈹? 缁熶竴鑴氭墜鏋?  鈹?    鍙鐜?             鈹?
              鈹?鎵€鏈?lesson   鈹?鍏变韩 dlhub/  鈹?绉嶅瓙 + 閰嶇疆 + 鏃ュ織      鈹?
              鈹?鏀寔绂荤嚎鍐掔儫   鈹?璁粌妗嗘灦      鈹?姣忔瀹為獙鍙拷婧?         鈹?
              鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
              鈹?  娓愯繘寮?     鈹? 娴嬭瘯瑕嗙洊     鈹? Model Zoo             鈹?
              鈹?鐢辨祬鍏ユ繁       鈹?126 pytest  鈹?2500+ 鏋舵瀯 ID          鈹?
              鈹?8 track 閫掕繘  鈹?CI 鍙泦鎴?   鈹?鍏ㄩ鍩熺粺涓€鎺ュ彛           鈹?
              鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
```

<details>
<summary><b>璇︾粏璇存槑</b></summary>

- **Offline-first** 鈥?鎵€鏈?lesson 鏀寔 `--dataset fake` 绂荤嚎鍐掔儫锛屾棤闇€涓嬭浇浠讳綍鏁版嵁闆嗭紝10 绉掑唴楠岃瘉鐜
- **缁熶竴鑴氭墜鏋?* 鈥?鎵€鏈?lesson 鍏变韩 `dlhub/` 妗嗘灦锛氳缁冨惊鐜€佽澶囩鐞嗐€佺瀛愩€佹鏌ョ偣銆丣SONL 鎸囨爣璁板綍
- **鍙鐜?* 鈥?绉嶅瓙绠＄悊 + 閰嶇疆鑷姩淇濆瓨 + 鎸囨爣鏃ュ織锛屾瘡娆″疄楠屽畬鏁村彲杩芥函
- **娓愯繘寮?* 鈥?浠庡熀纭€寮犻噺鎿嶄綔鍒?Vision Transformer銆丟raphSAGE銆丳ointNet++銆丩LaVA锛岀敱娴呭叆娣憋紝8 涓?track 灞傚眰閫掕繘
- **娴嬭瘯瑕嗙洊** 鈥?126 pytest 娴嬭瘯鏂囦欢瑕嗙洊妗嗘灦鏍稿績涓庢墍鏈?track锛屾敮鎸?CI 闆嗘垚
- **Model Zoo** 鈥?鍏ㄩ鍩燂紙Vision / NLP / Point Cloud / Multimodal / Generative / Federated锛夊叡 2500+ 鏋舵瀯 ID锛岀函 PyTorch 鏈湴瀹炵幇锛岀粺涓€鎺ュ彛涓€琛屽垏鎹?

</details>

---

## Contributing

娆㈣繋璐＄尞锛佹棤璁烘槸淇 typo銆佽ˉ鍏?lesson 杩樻槸鎻愬嚭鏂扮殑 track 鎯虫硶銆?

1. Fork 鏈粨搴?
2. 鍒涘缓浣犵殑鍒嗘敮 (`git checkout -b feature/amazing-lesson`)
3. 閬靛惊 [`docs/STYLEGUIDE.md`](docs/STYLEGUIDE.md) 浠ｇ爜瑙勮寖
4. 纭繚 `python scripts/smoke_check.py` 閫氳繃
5. 鎻愪氦 PR

> [!NOTE]
> 姣忎釜鏂?lesson 搴斿寘鍚細`model.py` / `data.py` / `train.py` / `README.md`锛屽苟鏀寔 `--dataset fake` 鍐掔儫妯″紡銆傝瑙?[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)銆?

---

## Citation

濡傛灉鏈」鐩浣犵殑瀛︿範鎴栫爺绌舵湁甯姪锛屾杩庡紩鐢細

```bibtex
@misc{dlhub2026,
  title  = {DL-Hub: A Unified PyTorch Deep Learning Learning Project},
  author = {DL-Hub Contributors},
  year   = {2026},
  url    = {https://github.com/your-username/DL-Hub}
}
```

---

## License

鏈」鐩噰鐢?[MIT License](LICENSE) 寮€婧愩€備唬鐮佽嚜鐢变娇鐢紝`resources/pdfs/` 涓嬬殑璁烘枃鐗堟潈褰掑師浣滆€呮墍鏈夈€?

---

<div align="center">

**Built for learning. Built to run.**

<sub>濡傛灉瑙夊緱鏈夊府鍔╋紝娆㈣繋 Star 鏀寔 猸?/sub>

</div>
