# README SVG 重构与完善设计

**日期**：2026-04-18
**作者**：skygazer42 <207829897@qq.com>
**状态**：Draft — 等待 spec reviewer + 用户审阅

---

## 1. 背景

当前 `README.md` 引用 3 张 PNG：

| 文件 | 大小 | 用途 |
|------|------|------|
| `assets/hero_banner.png` | 6.1 MB | 顶部 banner |
| `assets/overview_8panels.png` | 6.7 MB | "What You'll Build" 八领域示意 |
| `assets/learning_path_steps.png` | 6.1 MB | Learning Path 八步示意 |

问题：

- 三张图合计 **~19 MB**，占 git 体积大、无法 diff
- 非矢量，高清屏/缩放模糊
- 风格散、色调不统一，与 README 的 badge/卡片字号系统脱节
- "Design Philosophy" 章节仍是 ASCII 图，与其他装饰视觉落差明显
- README 其他章节（Quick Start / Learning Tracks / Model Zoo）缺少视觉导引，只有密集表格

## 2. 目标

- **三张 PNG 全部替换为手写 SVG**（矢量、可 diff、~几 KB）
- **新增 4 张 SVG**：Zoo 生态全貌图、Design Philosophy 矩阵、Track 体量条形、Quick Start 五步管道
- 统一**现代科技风**视觉（深色系 + PyTorch 橙 + 几何网格 + 轻微 glow）
- 每张图出 `-light.svg` / `-dark.svg` **双版本**，README 用 `<picture>` 自动切换
- README 配套润色：caption、段落衔接、去冗余文案

## 3. 非目标（YAGNI）

- 不改 README 的章节结构或导航目录（内容层级保持）
- 不重写 lesson 表格（内容信息密度已经够）
- 不删除 `assets/*.png`（保留避免外链破坏，只是 README 不再引用）
- 不引入动画 SVG / 交互 SVG（静态即可）
- 不搭建图片生成脚本或 CI 流水（手写一次就够）

## 4. 方案

### 4.1 实现方式

手写 SVG XML（方案 A）。理由：

- 现代科技风依赖渐变 / glow / 精细间距 / 字体，matplotlib / drawsvg 难做
- 每张图一年改动 <2 次，脚本化 ROI 低
- 手写的 SVG 可 git diff、可在 editor 里即时改
- 0 dev 依赖

排除 matplotlib 脚本方案（风格不够）和混合方案（两套工具链无必要）。

### 4.2 文件组织

```
assets/
├── hero_banner.png              # 保留不删，README 不再引用
├── overview_8panels.png
├── overview_4panels.png
├── learning_path_steps.png
└── svg/                         # 新增目录
    ├── hero-light.svg
    ├── hero-dark.svg
    ├── overview-light.svg
    ├── overview-dark.svg
    ├── learning-path-light.svg
    ├── learning-path-dark.svg
    ├── zoo-ecosystem-light.svg
    ├── zoo-ecosystem-dark.svg
    ├── philosophy-light.svg
    ├── philosophy-dark.svg
    ├── track-volume-light.svg
    ├── track-volume-dark.svg
    ├── pipeline-light.svg
    └── pipeline-dark.svg
```

### 4.3 Design Tokens

```
# Dark palette
bg-base      #0D1117
bg-elevated  #161B22
bg-gradient  linear-gradient #0A0E27 → #0D1117
fg-primary   #E6EDF3
fg-muted     #8B949E
stroke       #30363D
accent-orange #EE4C2C   (PyTorch)
accent-blue   #58A6FF   (cyan highlight)
accent-cyan   #39D0D8   (zoo / data)
accent-green  #2EA043   (pipeline ok)
accent-purple #A371F7   (LLM / multimodal)

# Light palette
bg-base      #FFFFFF
bg-elevated  #F6F8FA
fg-primary   #1F2328
fg-muted     #656D76
stroke       #D0D7DE
accent-orange #EE4C2C
accent-blue   #0969DA
accent-cyan   #1B7C83
accent-green  #1A7F37
accent-purple #8250DF
```

- 字体：`'Inter','Segoe UI',-apple-system,sans-serif` + 代码 `'JetBrains Mono','ui-monospace',monospace`
- 统一圆角：4 px（chip）/ 8 px（card）/ 12 px（panel）
- stroke 宽度：1 / 1.5 / 2 三档

### 4.4 主题切换 markup

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/svg/hero-dark.svg">
  <img src="assets/svg/hero-light.svg" alt="DL-Hub — 339 lessons · 8 tracks · 8000+ architectures" width="100%">
</picture>
```

GitHub README 渲染自动按用户主题切换。alt 文案保证可访问性。

## 5. 七张 SVG 布局设计

### 5.1 hero（1200×420）

顶部 banner。左大字标题 + 副标题，右 stat 列表，底部 8 个 track chip。

- 背景：`bg-gradient` + 低透明度几何点阵（12px 间距）
- 左：**DL-Hub**（64 px 粗体）+ 副标题两行（18 px）
- 右：4 个带图标条目 `339 lessons / 8 tracks / 8000+ arch / 393 tests`
- 底部：8 个 chip，`Vision / NLP / GNN / Point Cloud / Generative / Multimodal / LLM / Federated`，每个配一个 accent color

### 5.2 overview（1200×600）

4×2 网格，每格 260×240。

- 每格：顶部 24×24 几何 icon，再是 track 名（18 px）、lesson 数（14 px dim）、一句话说明（12 px）
- 8 个格子按 accent 轮换（orange→blue→cyan→green→purple→pink→yellow→red）
- 格间细线连接（1 px stroke），交点放 2px 发光圆点
- 底部小字："① Vision — ② NLP — … — ⑧ Federated"（保留原 caption 逻辑）

### 5.3 learning-path（1200×300）

横向 8 节点串珠。

- 编号圆（36 px）+ track 名（14 px）+ lesson 数 chip（12 px，accent 色底）
- lesson 数来自 `tracks/*/lesson_*` 实际目录数：`2 / 89 / 49 / 11 / 36 / 51 / 43 / 58`（合计 339）
- 节点间虚线 + 渐变箭头
- 顶部标注时间标尺：`2 min` → `2 days` → `2 weeks` → `6-8 weeks`
- 右端终点标记 `Full Curriculum ✓` 徽章

### 5.4 zoo-ecosystem（1200×760）

中心放射结构。

- 正中：圆形主节点 "DL-Hub Zoo / 8000+ arch ID"，半径 80 px
- 内圈（半径 180 px）：6 大区块节点 `Vision · NLP · Point Cloud · Multimodal · Generative · Federated`，每个带核心计数
- 外圈（半径 320 px）：21 个子 zoo 节点，按类别分扇区，标架构数
- 放射线从内圈连到子节点
- 四角 4 个 callout box：`Lazy Import` / `统一接口` / `纯 PyTorch` / `CLI 工具`

### 5.5 philosophy（1200×500）

2×3 矩阵替换当前 ASCII 6 格。

- 每格：左上 20×20 几何 icon，主标题（18 px），副描述（13 px）
- 六格内容完全沿用现有文案：Offline-first / 统一脚手架 / 可复现 / 渐进式 / 测试覆盖 / Model Zoo
- 格间横纵连线 + 交点发光
- 背景放 1% 透明度网格

### 5.6 track-volume（1100×500）

横向条形图。数据（源：`tracks/*/lesson_*` 实际目录数，合计 339）：

```
Vision      89
Multimodal  58
Generative  51
NLP         49
LLM         43
Point Cloud 36
GNN         11
Foundations 2
```

- 每条高度 40 px，间距 12 px
- 条左端 track 名 + lesson 数字，条色 accent，尾部标核心概念小字（如 "CNN / ViT / 检测 / 分割"）
- 最长条（Vision 89）走满 800 px，其余按比例
- 右上注 "Total · 339 lessons"（与 README 顶部 stat 口径一致）

### 5.7 pipeline（1200×240）

横向 5 步管道。

- 5 节点：`clone` → `install` → `smoke_check` → `run lesson` → `metrics`
- 每节点：40 px 方框含几何 icon + 步骤名（14 px）+ 命令片段 mono（11 px）
- 节点间箭头 + 渐变进度条
- 底部横标尺 "── 2 min on CPU ──▶"，右端绿色勾号

## 6. README 修改清单

| 位置 | 旧 | 新 |
|------|---|---|
| L3 hero | `<img src="assets/hero_banner.png">` | `<picture>` + hero svg 双版 |
| L79 overview | `<img src="assets/overview_8panels.png">` | `<picture>` + overview svg 双版 |
| L172 learning path | `<img src="assets/learning_path_steps.png">` | `<picture>` + learning-path svg 双版 |
| Quick Start 章节开头（L105 附近） | — | 插入 pipeline svg + 1 行 caption |
| What You'll Build 后 / Model Zoo 开头（L751 附近） | — | 插入 zoo-ecosystem svg + 1 行 caption |
| Design Philosophy 章节（L1559）| ASCII 6 格 | philosophy svg 双版 |
| Learning Tracks 章节开头（L210 附近）| — | 插入 track-volume svg + 1 行 caption |
| "课程及代码合集" 表格（L212-L219）| lesson 数为老数据（47/27/23/22/21/36 等，与顶部"339 Lessons"不符）| 同步修正为 `2 / 89 / 49 / 11 / 36 / 51 / 43 / 58`（与 SVG 一致，合计 339）|

同时润色：

- 为每张新图加 `<p align="center"><sub>...</sub></p>` 说明
- 若 "Design Philosophy" 下的 `<details>` 文字与新图冗余，保留 `<details>` 但缩短
- caption 用 **中文主**，英文术语点缀

## 7. 成功标准

- 所有 14 个 SVG 文件在 Chrome / Firefox / Safari / GitHub renderer 里正确渲染
- 深色 / 浅色主题切换无缝（GitHub 用户无需手动选图）
- 每张 SVG < 20 KB（基准：手写 XML 最多几 KB）
- README 在 GitHub 页面上首屏视觉有可辨识的 "DL-Hub 品牌调性"
- `git diff --stat` 显示 assets 增量合理（新增 14 份 SVG，合计 < 300 KB）
- README 纯表格区域无破坏，lesson 表格和 Zoo 表格完整

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 手写 SVG 字符量大、错位风险 | 分阶段实现，每张完工后独立在浏览器预览验证 |
| GitHub `<picture>` 在某些 Markdown 渲染器兼容性差 | 保留 `<img>` 作 fallback（`<source>` + `<img>` 结构原生支持 fallback） |
| 双版本 SVG 文案如果变更需要同步 2 处 | 命名保持对称 `*-light / *-dark`，PR 审查强制配对检查 |
| 字体在不支持 Inter 的系统上掉回系统字体，视觉略退化 | 系统栈 `Inter, Segoe UI, -apple-system, sans-serif` 已包含 fallback |
| 新图和 README 文字重复冗余 | 图与文相辅相成 — 图是 "快速扫视"，表/文字是 "查具体数字"，不冲突 |

## 9. 实现分阶段（由 writing-plans 进一步拆解）

1. **M1 Scaffolding**：建 `assets/svg/` 目录，写 Design Tokens 注释头（两主题共享）
2. **M2 三张替换图**：hero / overview / learning-path 双版本各 3 张 = 6 个文件，替换 README 引用
3. **M3 四张新增图**：zoo-ecosystem / philosophy / track-volume / pipeline 各 2 版 = 8 个文件
4. **M4 README 布线**：插入 4 张新图到对应章节，补 caption，删除 ASCII Design Philosophy 块
5. **M5 验收**：浏览器预览 14 张 SVG、GitHub 本地 preview（`grip` 或直接 push 到 draft 分支预览）、README 长度合理检查

每阶段可独立 commit，允许中途 rollback。
