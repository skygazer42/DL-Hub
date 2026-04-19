# README SVG 重构与完善 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 README 里 3 张 PNG 替换为 7 张手写 SVG（其中 4 张是新增），每张出深浅双主题，通过 `<picture>` 自动切换，同时修正 README L212-L219 过时的 lesson 数据表。

**Architecture:** 全部 SVG 手写 XML 到 `assets/svg/`，统一现代科技风 Design Tokens（深色 `#0D1117` / 浅色 `#FFFFFF` + PyTorch 橙 `#EE4C2C`）。用 `<picture>` + `prefers-color-scheme` 自动切换。验证走 `tests/test_readme_svg_assets.py`，用 `xml.etree.ElementTree` 校验 SVG 结构 + `re` 校验 README 引用完整性，无外部依赖。

**Tech Stack:** 纯 XML (SVG 1.1), Markdown, Python stdlib (xml.etree, re, pathlib) for validation, pytest

**Spec:** `docs/superpowers/specs/2026-04-18-readme-svg-refactor-design.md`

---

## Design Tokens（所有 SVG 共享）

```
# Dark
bg-base        #0D1117
bg-elevated    #161B22
bg-gradient-a  #0A0E27  (top-left)
bg-gradient-b  #0D1117  (bottom-right)
fg-primary     #E6EDF3
fg-muted       #8B949E
stroke         #30363D
accent-orange  #EE4C2C  (PyTorch)
accent-blue    #58A6FF
accent-cyan    #39D0D8
accent-green   #2EA043
accent-purple  #A371F7
accent-pink    #F778BA
accent-yellow  #D29922
accent-red     #F85149

# Light
bg-base        #FFFFFF
bg-elevated    #F6F8FA
fg-primary     #1F2328
fg-muted       #656D76
stroke         #D0D7DE
accent-orange  #EE4C2C
accent-blue    #0969DA
accent-cyan    #1B7C83
accent-green   #1A7F37
accent-purple  #8250DF
accent-pink    #BF3989
accent-yellow  #9A6700
accent-red     #CF222E

# Typography
font-sans  'Inter','Segoe UI',-apple-system,sans-serif
font-mono  'JetBrains Mono','ui-monospace',monospace

# Radii: 4/8/12 px, Strokes: 1/1.5/2 px
```

## File Structure

**Create:**
```
assets/svg/
├── hero-light.svg           hero-dark.svg
├── overview-light.svg       overview-dark.svg
├── learning-path-light.svg  learning-path-dark.svg
├── zoo-ecosystem-light.svg  zoo-ecosystem-dark.svg
├── philosophy-light.svg     philosophy-dark.svg
├── track-volume-light.svg   track-volume-dark.svg
└── pipeline-light.svg       pipeline-dark.svg

tests/test_readme_svg_assets.py    # 验证 SVG + README 引用
```

**Modify:**
```
README.md    # L3, L79, L172 替换 + 4 处插入 + L212-L219 修数 + Design Philosophy 替换
```

**Keep as-is:**
```
assets/hero_banner.png             # 保留避免外链破坏
assets/overview_8panels.png
assets/overview_4panels.png
assets/learning_path_steps.png
```

---

## Task 1: 测试骨架 + scaffolding

**Files:**
- Create: `assets/svg/.gitkeep`
- Create: `tests/test_readme_svg_assets.py`

- [ ] **Step 1: 写失败测试（全集）**

Create `tests/test_readme_svg_assets.py`:

```python
"""Validate README SVG refactor artifacts.

Tests:
1. Each of 14 SVG files parses as valid XML with <svg> root + viewBox.
2. Each SVG has <title> child for accessibility.
3. Each SVG byte-size is <= 50 KB.
4. README.md references all 7 image pairs via <picture> (no bare PNG <img> for
   hero/overview/learning-path).
5. README.md no longer embeds the ASCII philosophy box (replaced by svg).
6. README.md L212-L219 track table lesson counts match tracks/*/lesson_* dir counts.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SVG_DIR = REPO / "assets" / "svg"
README = REPO / "README.md"
TRACKS = REPO / "tracks"

SVG_BASES = [
    "hero",
    "overview",
    "learning-path",
    "zoo-ecosystem",
    "philosophy",
    "track-volume",
    "pipeline",
]
THEMES = ("light", "dark")
SVG_NS = "{http://www.w3.org/2000/svg}"
MAX_BYTES = 50 * 1024


@pytest.mark.parametrize("base", SVG_BASES)
@pytest.mark.parametrize("theme", THEMES)
def test_svg_parses(base: str, theme: str) -> None:
    path = SVG_DIR / f"{base}-{theme}.svg"
    assert path.exists(), f"missing {path.relative_to(REPO)}"
    assert path.stat().st_size <= MAX_BYTES, (
        f"{path.name} is {path.stat().st_size} B, exceeds {MAX_BYTES}"
    )
    root = ET.parse(path).getroot()
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("viewBox"), f"{path.name} missing viewBox"
    title = root.find(f"{SVG_NS}title")
    assert title is not None and title.text, f"{path.name} missing <title>"


def test_readme_uses_picture_for_replaced_banners() -> None:
    text = README.read_text(encoding="utf-8")
    # Old PNG <img> references should be gone for the three replaced banners.
    for old in ("assets/hero_banner.png",
                "assets/overview_8panels.png",
                "assets/learning_path_steps.png"):
        assert old not in text, f"README still references {old}"
    # Expect at least 7 <picture> blocks.
    assert text.count("<picture>") >= 7, "expected >= 7 <picture> blocks"


def test_readme_no_ascii_philosophy_box() -> None:
    text = README.read_text(encoding="utf-8")
    assert "DL-Hub 设计理念" not in text, (
        "ASCII philosophy box should be replaced by philosophy SVG"
    )


def test_readme_track_table_counts_match_dir() -> None:
    """L212-L219 `课程及代码合集` row数字应与 tracks/*/lesson_* 目录数一致。"""
    text = README.read_text(encoding="utf-8")
    expected = {
        "Foundations": len(list((TRACKS / "foundations").glob("lesson_*"))),
        "Vision": len(list((TRACKS / "vision").glob("lesson_*"))),
        "NLP": len(list((TRACKS / "nlp").glob("lesson_*"))),
        "GNN": len(list((TRACKS / "gnn").glob("lesson_*"))),
        "Point Cloud": len(list((TRACKS / "pointcloud").glob("lesson_*"))),
        "Generative": len(list((TRACKS / "generative").glob("lesson_*"))),
        "LLM": len(list((TRACKS / "llm").glob("lesson_*"))),
        "Multimodal": len(list((TRACKS / "multimodal").glob("lesson_*"))),
    }
    for name, count in expected.items():
        pattern = (
            rf"<b>{re.escape(name)}</b><br/><sub>{count} lessons</sub>"
        )
        assert re.search(pattern, text), (
            f"expected table cell '{name} — {count} lessons', not found"
        )
```

- [ ] **Step 2: 验证失败**

Run: `cd /data/temp42/DL-Hub && pytest tests/test_readme_svg_assets.py -q`
Expected: 失败。14 个 parametrize 的测试缺文件，`<picture>` 计数 <7，ASCII philosophy 仍在，L212-L219 数字不对。

- [ ] **Step 3: Scaffolding**

Create empty placeholder:
```bash
mkdir -p /data/temp42/DL-Hub/assets/svg && touch /data/temp42/DL-Hub/assets/svg/.gitkeep
```

- [ ] **Step 4: 提交骨架**

```bash
cd /data/temp42/DL-Hub
git add tests/test_readme_svg_assets.py assets/svg/.gitkeep
git commit -m "test(readme): add SVG & README reference validation harness"
```

---

## Task 2: hero-{light,dark}.svg

**Files:**
- Create: `assets/svg/hero-light.svg`
- Create: `assets/svg/hero-dark.svg`

**Spec ref:** §5.1 — 1200×420 顶部 banner

**SVG 结构要求：**
- `viewBox="0 0 1200 420"` `xmlns="http://www.w3.org/2000/svg"`
- `<title>DL-Hub — Deep Learning from Scratch</title>`
- `<defs>` 含 `bg-gradient`（linear `#0A0E27 → #0D1117` dark / `#FFFFFF → #F6F8FA` light）
- 背景层：gradient rect + 12 px 点阵（`<pattern>` 定义 dot，用 `fill`）
- 左侧 `(48, 120)` 起 "DL-Hub" 大标题：`font-size="88"` `font-weight="700"` `fill="accent-orange"`
- 标题下方 "从零手写 · 循序渐进 — PyTorch 深度学习统一学习项目"：`font-size="20"` `fill="fg-primary"`
- 再下方 "Hand-written · Progressive · Offline-first"：`font-size="14"` `fill="fg-muted"`
- 右侧 `(760, 120)` 起 4 行 stat：`339 lessons` / `8 tracks` / `8000+ architectures` / `393 tests`，每行左侧 12 px 圆点（依次 orange / blue / cyan / green）
- 底部 `y=340` 8 个 chip（45×32 圆角矩形）横向排列：Vision / NLP / GNN / Point Cloud / Generative / Multimodal / LLM / Federated，chip 底色 `bg-elevated` + 1 px stroke，文字 `font-size="12"`，chip 边框用对应 accent 色
- 所有文字 `font-family` 用 `'Inter',...` 系统回退栈

- [ ] **Step 1: 写 hero-dark.svg**

完整手写深色版 SVG，遵循上面结构。写完后运行自动 XML 校验：

```bash
python -c "import xml.etree.ElementTree as ET; ET.parse('/data/temp42/DL-Hub/assets/svg/hero-dark.svg')"
```

- [ ] **Step 2: 写 hero-light.svg**

复制 hero-dark 结构，替换所有 token：`#0D1117→#FFFFFF`、`#0A0E27→#F6F8FA`、`#E6EDF3→#1F2328`、`#8B949E→#656D76`、`#30363D→#D0D7DE`、accent-blue `#58A6FF→#0969DA`、accent-cyan `#39D0D8→#1B7C83`、accent-green `#2EA043→#1A7F37`。橙色 `#EE4C2C` 两主题共用。

- [ ] **Step 3: 运行相关测试**

```bash
cd /data/temp42/DL-Hub
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k hero
```
Expected: 两条 PASS（light + dark）。

- [ ] **Step 4: 浏览器人眼预览（可选但推荐）**

提示用户在浏览器打开 `file:///data/temp42/DL-Hub/assets/svg/hero-dark.svg` 目视检查。

- [ ] **Step 5: Commit**

```bash
git add assets/svg/hero-light.svg assets/svg/hero-dark.svg
git commit -m "feat(assets): hero banner SVG (light + dark)"
```

---

## Task 3: overview-{light,dark}.svg

**Files:**
- Create: `assets/svg/overview-light.svg`
- Create: `assets/svg/overview-dark.svg`

**Spec ref:** §5.2 — 1200×600 八领域 4×2 网格

**结构要求：**
- `viewBox="0 0 1200 600"` `<title>DL-Hub 八大领域</title>`
- 背景：全幅 `bg-base` + 1% 透明度 32 px 网格 `<pattern>`
- 4×2 网格，每格 280×260，外边距 20 px，格间距 20 px
- 每格：rounded rect 12 px radius，1 px stroke，accent 色左上 24×24 几何 icon，18 px 粗体 track 名，12 px `fg-muted` lesson count，13 px 两行 track 核心描述（见下方文案）
- 八格 accent 轮换：Vision=orange, NLP=blue, GNN=cyan, Point Cloud=green, Generative=purple, Multimodal=pink, LLM=yellow, Federated=red
- icon 几何形：Vision=眼形（椭圆+圆）、NLP=「字形、GNN=三圆连线、Point Cloud=散点 9 个、Generative=波浪两条、Multimodal=两方叠、LLM=方块堆叠、Federated=中心辐射
- 格间八条细线（`stroke-width=1` `stroke-dasharray="2,2"`）连接邻格中点，交点放 2 px 发光圆

**8 格文案**（顺序 = README What You'll Build 表格）：
| 格 | 标题 | lesson | 描述 |
|---|------|--------|------|
| 1 | Vision | 89 lessons | 从 LeNet 到 ViT · 分类/检测/分割 |
| 2 | NLP | 49 lessons | 词嵌入到 Transformer · 分类/NER/阅读理解 |
| 3 | GNN | 11 lessons | GCN 到 PinSAGE · 节点/图/推荐 |
| 4 | Point Cloud | 36 lessons | PointNet 到 PCT · 3D 分类/分割/补全 |
| 5 | Generative | 51 lessons | VAE/GAN/Diffusion/Flow · 重建与生成 |
| 6 | Multimodal | 58 lessons | CLIP 到 Audio-Visual · VLM/检索 |
| 7 | LLM | 43 lessons | Causal LM / SFT / RLHF · 50+ 论文笔记 |
| 8 | Federated | 76 strategies | FedAvg 到 DP · 隐私/个性化/安全聚合 |

> ⚠ 第 8 格"Federated"显示策略数（76 families）而非 lesson 数，与 README 联邦学习 Zoo 一致。

- [ ] **Step 1: 写 overview-dark.svg**
- [ ] **Step 2: 写 overview-light.svg**
- [ ] **Step 3: 测试**
```bash
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k overview
```
- [ ] **Step 4: Commit**
```bash
git add assets/svg/overview-light.svg assets/svg/overview-dark.svg
git commit -m "feat(assets): 8-domain overview SVG (light + dark)"
```

---

## Task 4: learning-path-{light,dark}.svg

**Files:**
- Create: `assets/svg/learning-path-light.svg`
- Create: `assets/svg/learning-path-dark.svg`

**Spec ref:** §5.3 — 1200×300 横向 8 节点串珠

**结构要求：**
- `viewBox="0 0 1200 300"` `<title>8 Learning Tracks Progression</title>`
- 背景全幅 bg-base
- 顶部 y=40 时间标尺：4 个刻度 `2 min` / `2 days` / `2 weeks` / `6-8 weeks`
- 节点中心 y=150，8 个节点间距 (1200-96) / 7 ≈ 157 px，起点 x=72
- 每节点：36 px 圆，accent 色填充（Foundations=橙、Vision=橙、NLP=蓝、GNN=青、PC=绿、Gen=紫、LLM=黄、MM=粉）
  - 圆内：白字 step 号 1-8
  - 圆下 y=210：track 名（15 px 粗体）
  - track 名下 y=234：lesson 数 chip（"89 lessons" 等），chip 底 bg-elevated，字 `font-size=11`
- 节点间横向虚线连接，`stroke-dasharray="4,4"`，渐变色从左到右按 accent 依次过渡
- 右端 x=1160, y=150 "Full Curriculum ✓" 徽章（绿 chip）

**lesson 数：2 / 89 / 49 / 11 / 36 / 51 / 43 / 58**（来自 spec §5.3 修正后）

- [ ] **Step 1: 写 learning-path-dark.svg**
- [ ] **Step 2: 写 learning-path-light.svg**
- [ ] **Step 3: 测试**
```bash
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k learning-path
```
- [ ] **Step 4: Commit**
```bash
git add assets/svg/learning-path-light.svg assets/svg/learning-path-dark.svg
git commit -m "feat(assets): 8-step learning path SVG (light + dark)"
```

---

## Task 5: zoo-ecosystem-{light,dark}.svg

**Files:**
- Create: `assets/svg/zoo-ecosystem-light.svg`
- Create: `assets/svg/zoo-ecosystem-dark.svg`

**Spec ref:** §5.4 — 1200×760 中心放射

**结构要求：**
- `viewBox="0 0 1200 760"` `<title>DL-Hub Zoo Ecosystem</title>`
- 中心 (600, 380)：主圆 r=80，accent-orange 填充径向渐变，白字 "DL-Hub Zoo" + "8000+ arch"
- 内圈（r=200 轨道）6 节点，每节点 r=42，均匀分布 60° 一个：
  - Vision (top, 12 点钟方向)、NLP (2 点)、Point Cloud (4 点)、Multimodal (6 点)、Generative (8 点)、Federated (10 点)
  - 每节点 accent 色 + 白字，内含 track 名 + 核心计数（Vision=791 arch / NLP=813 arch / PC=64 arch / MM=70 families / Gen=76 arch / Fed=76 strategies）
- 外圈（r=340）**22 个小节点**（与 README L757 Zoo 子系统总览表格行数一致；README 标题旧文案"21 个"保持不改），r=22，分布在对应大区扇区内：
  - Vision 扇（9 个）：Backbones, Detection (2D), InsSeg, PanSeg, Lane, CoSeg, FGVC, ActionRec, MOT
  - NLP 扇（1 个）：Text Encoders
  - PC 扇（6 个）：Backbones, 3D Det, 3D Seg, 3D InsSeg, 3D Track, 3DGS
  - MM 扇（2 个）：VLM, Prompt Learning
  - Vision 扇扩展（1 个）：New Directions Batch XIII
  - Gen 扇（2 个）：GAN, Diffusion
  - Fed 扇（1 个）：FL Strategies
- 放射线从中心穿过内圈连到每个外圈节点，虚线 `stroke-dasharray="2,3"`
- 四角 callout box（120×60 圆角矩形 bg-elevated + stroke）：
  - 左上：`Lazy Import` 副标 "0 startup cost"
  - 右上：`统一接口` 副标 "build(arch_id)"
  - 左下：`纯 PyTorch` 副标 "no pretrain needed"
  - 右下：`CLI 工具` 副标 "--list/--search/--smoke"

- 因 22 个节点文字密集，标签可以简写。必要时把小 zoo 节点改成数字+短码（如 "Det / 140"），hover 在 GitHub 上不支持，所以文字要直接可读。

- [ ] **Step 1: 写 zoo-ecosystem-dark.svg**
- [ ] **Step 2: 写 zoo-ecosystem-light.svg**
- [ ] **Step 3: 测试**
```bash
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k zoo-ecosystem
```
- [ ] **Step 4: Commit**
```bash
git add assets/svg/zoo-ecosystem-light.svg assets/svg/zoo-ecosystem-dark.svg
git commit -m "feat(assets): Zoo ecosystem radial SVG (light + dark)"
```

---

## Task 6: philosophy-{light,dark}.svg

**Files:**
- Create: `assets/svg/philosophy-light.svg`
- Create: `assets/svg/philosophy-dark.svg`

**Spec ref:** §5.5 — 1200×500 2×3 矩阵

**结构要求：**
- `viewBox="0 0 1200 500"` `<title>DL-Hub Design Philosophy</title>`
- 2×3 网格：每格 370×220，外边距 30 px，格间距 20 px
- 每格：rounded rect 12 px + 1 px stroke，左上 28×28 几何 icon，18 px 粗体标题，13 px 两行描述
- 6 格（顺序同 README 原 ASCII）：
  | 位置 | 标题 | 描述 |
  |---|---|---|
  | (0,0) | Offline-first | 所有 lesson 支持 `--dataset fake` 离线冒烟<br/>无需下载数据集，10 秒验证环境 |
  | (0,1) | 统一脚手架 | 所有 lesson 共享 `dlhub/` 框架<br/>训练循环/设备/种子/检查点 |
  | (0,2) | 可复现 | 种子 + 配置 + JSONL 指标<br/>每次实验完整可追溯 |
  | (1,0) | 渐进式 | 8 个 track 层层递进<br/>从张量到 ViT/PointNet++/LLaVA |
  | (1,1) | 测试覆盖 | 393 pytest 测试<br/>覆盖框架核心与所有 track |
  | (1,2) | Model Zoo | 8000+ 架构 ID<br/>纯 PyTorch 统一接口 |
- icon 几何形：Offline=云+斜杠、Scaffold=三层堆叠、Repro=循环箭头、Progressive=上升阶梯、Testing=勾号框、Zoo=辐射网
- 格间横纵 2 条连线（十字交错），交点 4 px 发光圆

- [ ] **Step 1: 写 philosophy-dark.svg**
- [ ] **Step 2: 写 philosophy-light.svg**
- [ ] **Step 3: 测试**
```bash
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k philosophy
```
- [ ] **Step 4: Commit**
```bash
git add assets/svg/philosophy-light.svg assets/svg/philosophy-dark.svg
git commit -m "feat(assets): design philosophy matrix SVG (light + dark)"
```

---

## Task 7: track-volume-{light,dark}.svg

**Files:**
- Create: `assets/svg/track-volume-light.svg`
- Create: `assets/svg/track-volume-dark.svg`

**Spec ref:** §5.6 — 1100×500 横向条形

**结构要求：**
- `viewBox="0 0 1100 500"` `<title>Lessons per Track</title>`
- 顶部标题 y=40：`8 Tracks · 339 Lessons` 20 px 粗体
- 右上 y=40 x=1060 小字 `Source: tracks/*/lesson_*`
- 8 条横向柱，每条高 40 px 间距 12 px，起始 y=80
- 每条布局：
  - x=0~160：track 名（14 px 粗体）
  - x=160~220：lesson 数数字（20 px）
  - x=240~：彩条（accent 色 + 2 px stroke），长度 = `count / max_count * 800`
  - 条末 x=1060：代表概念小字（12 px mono，fg-muted）
- **数据（按数量降序）：**
  | track | count | accent | 代表概念 |
  |---|---|---|---|
  | Vision | 89 | orange | CNN · ViT · 检测 · 分割 · 视频 |
  | Multimodal | 58 | pink | CLIP · VLM · 音视 · HOI |
  | Generative | 51 | purple | VAE · GAN · Diffusion · Flow |
  | NLP | 49 | blue | Transformer · NER · RAG |
  | LLM | 43 | yellow | SFT · DPO · RLHF · RAG |
  | Point Cloud | 36 | green | PointNet · DGCNN · 3D 分割 |
  | GNN | 11 | cyan | GCN · GAT · GraphSAGE |
  | Foundations | 2 | red | Autograd · Linear Regression |

- [ ] **Step 1: 写 track-volume-dark.svg**
- [ ] **Step 2: 写 track-volume-light.svg**
- [ ] **Step 3: 测试**
```bash
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k track-volume
```
- [ ] **Step 4: Commit**
```bash
git add assets/svg/track-volume-light.svg assets/svg/track-volume-dark.svg
git commit -m "feat(assets): track-volume horizontal bar SVG (light + dark)"
```

---

## Task 8: pipeline-{light,dark}.svg

**Files:**
- Create: `assets/svg/pipeline-light.svg`
- Create: `assets/svg/pipeline-dark.svg`

**Spec ref:** §5.7 — 1200×240 五步管道

**结构要求：**
- `viewBox="0 0 1200 240"` `<title>Quick Start Pipeline</title>`
- 5 个节点横向排列，中心 y=110，节点间距 (1200-200)/4 = 250 px，起点 x=100
- 每节点：70×70 圆角 rect（radius 12），bg-elevated 底，2 px 左边 accent 色条
  - 上方 y=60：序号圆（r=14，accent 填充，白字 1-5）
  - 节点中心 40 px icon（线性图标 stroke 2 px）
  - 节点下方 y=170：步骤名（14 px 粗体）
  - 步骤名下 y=190：命令片段（11 px mono `fg-muted`）
- **5 步：**
  | 序 | 图标 | 步骤 | 命令 | accent |
  |---|---|---|---|---|
  | 1 | git 分叉 | Clone | `git clone DL-Hub` | orange |
  | 2 | 向下箭头+包 | Install | `pip install -r requirements.txt` | blue |
  | 3 | 勾号盾 | Smoke | `python scripts/smoke_check.py` | cyan |
  | 4 | 播放三角 | Run Lesson | `python -m tracks.vision.lesson_01_mnist_lenet.train --dataset fake` | green |
  | 5 | 柱状图 | Metrics | `outputs/**/metrics.jsonl` | purple |
- 节点间右箭头（→），连接线底部 y=140 虚线
- 底部 y=220 标尺 `── 2 min end-to-end on CPU ──▶`，右端绿色勾号

- [ ] **Step 1: 写 pipeline-dark.svg**
- [ ] **Step 2: 写 pipeline-light.svg**
- [ ] **Step 3: 测试**
```bash
pytest tests/test_readme_svg_assets.py::test_svg_parses -q -k pipeline
```
- [ ] **Step 4: Commit**
```bash
git add assets/svg/pipeline-light.svg assets/svg/pipeline-dark.svg
git commit -m "feat(assets): quick-start 5-step pipeline SVG (light + dark)"
```

---

## Task 9: README — 替换 3 张 PNG 为 `<picture>`

**Files:**
- Modify: `README.md:3` (hero)
- Modify: `README.md:79` (overview)
- Modify: `README.md:172` (learning path)

**统一的 `<picture>` 模板：**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/svg/BASE-dark.svg">
  <img src="assets/svg/BASE-light.svg" alt="ALT" width="WIDTH" />
</picture>
```

- [ ] **Step 1: 替换 L3 hero**

Old:
```html
<img src="assets/hero_banner.png" width="100%" alt="DL-Hub — Deep Learning from Scratch" />
```

New:
```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/svg/hero-dark.svg">
  <img src="assets/svg/hero-light.svg" width="100%" alt="DL-Hub — 339 lessons · 8 tracks · 8000+ architectures · 393 tests" />
</picture>
```

- [ ] **Step 2: 替换 L79 overview**

Old:
```html
<img src="assets/overview_8panels.png" width="80%" alt="DL-Hub 八大领域：Vision · NLP · GNN · Point Cloud · Generative · Multimodal · LLM · Federated" />
```

New:
```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/svg/overview-dark.svg">
  <img src="assets/svg/overview-light.svg" width="90%" alt="DL-Hub 八大领域：Vision · NLP · GNN · Point Cloud · Generative · Multimodal · LLM · Federated" />
</picture>
```

- [ ] **Step 3: 替换 L172 learning-path**

Old:
```html
<img src="assets/learning_path_steps.png" width="85%" alt="8 Learning Tracks: Foundations → Vision → NLP → GNN → Point Cloud → Generative → LLM → Multimodal" />
```

New:
```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/svg/learning-path-dark.svg">
  <img src="assets/svg/learning-path-light.svg" width="92%" alt="8 Learning Tracks: Foundations → Vision → NLP → GNN → Point Cloud → Generative → LLM → Multimodal" />
</picture>
```

- [ ] **Step 4: 测试**

```bash
pytest tests/test_readme_svg_assets.py::test_readme_uses_picture_for_replaced_banners -q
```
Expected: PASS（3 个 PNG 路径已不在，且 `<picture>` 计数 >= 3）

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): replace 3 PNG banners with <picture>+SVG"
```

---

## Task 10: README — 插入 pipeline + zoo-ecosystem

**Files:**
- Modify: `README.md` — Quick Start 开头（`## Quick Start` 下 `> [!TIP]` 前）
- Modify: `README.md` — Model Zoo 开头（`## Model Zoo` 下 `> 全领域统一模型动物园` 后）

- [ ] **Step 1: Quick Start 前插入 pipeline**

在 `## Quick Start` 行下方（原 `> [!TIP]` 前）插入：

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/svg/pipeline-dark.svg">
    <img src="assets/svg/pipeline-light.svg" width="88%" alt="Quick Start: clone → install → smoke → run lesson → view metrics, 2 min end-to-end" />
  </picture>
</p>
<p align="center"><sub>五步跑通：clone → install → smoke → run lesson → metrics，CPU 上约 2 分钟完成全链路</sub></p>

```

- [ ] **Step 2: Model Zoo 开头插入 zoo-ecosystem**

在 `## Model Zoo` 下 `> 全领域统一模型动物园 — ...` 之后、`### Zoo 子系统总览（21 个子系统）` 之前插入：

```html

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/svg/zoo-ecosystem-dark.svg">
    <img src="assets/svg/zoo-ecosystem-light.svg" width="92%" alt="DL-Hub Zoo Ecosystem: 6 domains, 21 subsystems, 8000+ architectures" />
  </picture>
</p>
<p align="center"><sub>6 大领域 · 21 个 Zoo 子系统 · 8000+ 架构 ID，一行 <code>build(arch_id)</code> 切换</sub></p>

```

- [ ] **Step 3: 测试**

```bash
pytest tests/test_readme_svg_assets.py::test_readme_uses_picture_for_replaced_banners -q
```
Expected: PASS（`<picture>` 总数 ≥ 5，含 hero/overview/learning-path/pipeline/zoo-ecosystem）

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): insert pipeline + zoo-ecosystem SVG"
```

---

## Task 11: README — 替换 Design Philosophy ASCII + 插入 track-volume

**Files:**
- Modify: `README.md:~1559-1575` (Design Philosophy `<div>` 区块)
- Modify: `README.md:~208` (Learning Tracks 章节开头，"## 课程及代码合集" 前)

- [ ] **Step 1: 替换 Design Philosophy 的 ASCII 框**

把 `## Design Philosophy` 行后、`<details>` 行前的整个 ASCII 代码块（包括 \`\`\`...\`\`\` fence 内的 8 行 "DL-Hub 设计理念" 表）替换为：

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/svg/philosophy-dark.svg">
    <img src="assets/svg/philosophy-light.svg" width="92%" alt="DL-Hub Design Philosophy: Offline-first · 统一脚手架 · 可复现 · 渐进式 · 测试覆盖 · Model Zoo" />
  </picture>
</p>

```

`<details>` 及其后面详细说明原样保留。

- [ ] **Step 2: 在 "课程及代码合集" 前插入 track-volume**

找到 `## 课程及代码合集` 这一行，在**其前面**插入：

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/svg/track-volume-dark.svg">
    <img src="assets/svg/track-volume-light.svg" width="88%" alt="Lessons per track: Vision 89 · Multimodal 58 · Generative 51 · NLP 49 · LLM 43 · Point Cloud 36 · GNN 11 · Foundations 2" />
  </picture>
</p>
<p align="center"><sub>8 Tracks · 339 Lessons · 按 lesson 数量降序排列</sub></p>

```

- [ ] **Step 3: 测试 philosophy 消除 + picture 总数**

```bash
pytest tests/test_readme_svg_assets.py::test_readme_no_ascii_philosophy_box tests/test_readme_svg_assets.py::test_readme_uses_picture_for_replaced_banners -q
```
Expected: 两条 PASS。

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): replace philosophy ASCII + insert track-volume SVG"
```

---

## Task 12: README — 修正 "课程及代码合集" 表格 lesson 数

**Files:**
- Modify: `README.md`（在 Task 10 插入 pipeline 后实际行号会下移，用字符串搜替换即可，不依赖行号）

- [ ] **Step 1: 修改 6 处 lesson 数字**

以 `tracks/*/lesson_*` 实际目录数为准。用字符串替换（找到下列 old→new），顺序无关：

| old | new |
|-----|-----|
| `<b>Vision</b><br/><sub>47 lessons</sub>` | `<b>Vision</b><br/><sub>89 lessons</sub>` |
| `<b>NLP</b><br/><sub>27 lessons</sub>` | `<b>NLP</b><br/><sub>49 lessons</sub>` |
| `<b>Point Cloud</b><br/><sub>23 lessons</sub>` | `<b>Point Cloud</b><br/><sub>36 lessons</sub>` |
| `<b>Generative</b><br/><sub>22 lessons</sub>` | `<b>Generative</b><br/><sub>51 lessons</sub>` |
| `<b>LLM</b><br/><sub>21 lessons</sub>` | `<b>LLM</b><br/><sub>43 lessons</sub>` |
| `<b>Multimodal</b><br/><sub>36 lessons</sub>` | `<b>Multimodal</b><br/><sub>58 lessons</sub>` |

`Foundations` (2) 和 `GNN` (11) 已经正确，不改。

- [ ] **Step 2: 测试**

```bash
pytest tests/test_readme_svg_assets.py::test_readme_track_table_counts_match_dir -q
```
Expected: PASS。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): sync 课程及代码合集 lesson counts with tracks/ dir"
```

---

## Task 13: 全量验收

- [ ] **Step 1: 跑完整测试**

```bash
cd /data/temp42/DL-Hub
pytest tests/test_readme_svg_assets.py -v
```
Expected: 全部 PASS（14 个 parametrized + 3 个单测 = 17 passed）

- [ ] **Step 2: 检查 SVG 文件体积**

```bash
ls -lh assets/svg/*.svg | awk '{print $5, $9}'
```
Expected: 每个文件 < 50 KB（测试已断言）。目视检查合计 < 300 KB。

- [ ] **Step 3: 检查 README 长度变化**

```bash
wc -l README.md
git diff main --stat -- README.md
```
目的：确认 README 没有意外爆增。插入约 40 行 `<picture>` 块、删除 10 行 ASCII，净 +30 行在合理范围。

- [ ] **Step 4: 目视检查 README 在 GitHub 风格渲染下是否正常**

提示用户：
> 打开 `https://github.com/jhlu/DL-Hub` 或 `grip README.md` 本地预览，确认 7 张 SVG 显示正常，在浅色/深色主题下均可读。

- [ ] **Step 5: 向用户汇报完成**

列出所有 commit、文件变更统计、SVG 文件大小列表。

---

## 回滚策略

每个 SVG / README 改动独立 commit，按需 `git revert <sha>` 单独回退。测试 harness 是首个 commit，最后一步可选保留（作为后续改 README 的回归测试）或删除。

## Notes

- 所有 SVG 手写时避免用 `<style>` 元素（GitHub 可能剥离），改用 inline `fill=` / `stroke=` 属性
- `<title>` 元素必不可少（测试断言 + 可访问性）
- 字体全部走系统回退栈（Inter → Segoe UI → -apple-system → sans-serif），不引用外部字体文件
- 避免 SVG filter（如 `<filter>` drop-shadow），GitHub 有时渲染异常；需要发光效果时用半透明多重描边模拟
- `<picture>` 的 `<img>` fallback 放 **light 版本** 作为默认，和 GitHub 默认浅色主题一致
