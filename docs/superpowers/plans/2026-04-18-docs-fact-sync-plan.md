# DL-Hub Docs Fact Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every outdated numeric statistic in `README.md` and `docs/` with the 2026-04-18 measured ground truth, without touching scope, structure, or narrative text.

**Architecture:** Pure text edits only. No scripts, no CI, no new files (except one appended changelog block). All edits land in a single commit so revert is one command.

**Tech Stack:** Markdown only. Verification uses `grep` and optionally `mkdocs build --strict` (requirements-docs.txt).

**Spec reference:** `docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md`

---

## Ground Truth (measured 2026-04-18)

| Symbol | New value |
|---|---|
| `{lessons_total}` | 339 |
| `{tests_total}` | 393 |
| `{ml_algos}` | 31 |
| `{archs_total}` | 8000+ |
| `{vision_backbone_archs}` | 791 |
| `{nlp_backbone_archs}` | 814 |
| `{pointcloud_backbone_archs}` | 64 |
| `{pointcloud_lessons}` | 36 |

---

## Task 1: Re-measure & freeze ground truth

**Files:**
- Modify: none (verification only)

- [ ] **Step 1: Re-run the lesson count**

Run:
```bash
find tracks -maxdepth 2 -mindepth 2 -type d -name "lesson_*" | wc -l
```
Expected output: `339`

- [ ] **Step 2: Re-run the test-file count**

Run:
```bash
find tests -name "test_*.py" -type f | wc -l
```
Expected output: `393`

- [ ] **Step 3: Re-run the ML algorithm count**

Run:
```bash
ls ml_algorithms/python/*.py | grep -v __init__ | wc -l
```
Expected output: `31`

- [ ] **Step 4: Re-run the pointcloud lesson count**

Run:
```bash
find tracks/pointcloud -maxdepth 1 -mindepth 1 -type d -name "lesson_*" | wc -l
```
Expected output: `36`

- [ ] **Step 5: Re-run the per-zoo architecture counts**

Run (in project root, Python env with torch):
```bash
python - <<'PY'
import importlib
from pathlib import Path
from collections import defaultdict
bucket = defaultdict(int)
per_mod = {}
for p in Path('dlhub').rglob('*_zoo.py'):
    mod = str(p.with_suffix('')).replace('/', '.')
    m = importlib.import_module(mod)
    for fn in ('list_local_arches', 'list_local_archs'):
        if hasattr(m, fn):
            n = len(getattr(m, fn)())
            bucket[p.parts[1] if len(p.parts) >= 2 else 'root'] += n
            per_mod[mod] = n
            break
print('TOTAL =', sum(bucket.values()))
print('vision/local_zoo =', per_mod.get('dlhub.vision.local_zoo'))
print('nlp/local_zoo =', per_mod.get('dlhub.nlp.local_zoo'))
print('pointcloud/local_zoo =', per_mod.get('dlhub.pointcloud.local_zoo'))
PY
```
Expected output:
```
TOTAL = 8545
vision/local_zoo = 791
nlp/local_zoo = 814
pointcloud/local_zoo = 64
```

- [ ] **Step 6: Stop if any measured value differs from the table above**

If any mismatch, open `docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md`, update §3 to the new value, commit that spec patch, then redo subsequent tasks with the updated value. Do NOT proceed with stale numbers.

---

## Task 2: Edit `README.md`

**Files:**
- Modify: `README.md:18,36,41,51,333,468,753,1555,1568`

- [ ] **Step 1: Header badges (line 18)**

Replace the exact line:
```html
<code>339 Lessons</code> · <code>8 Learning Tracks</code> · <code>27 ML Algorithms</code> · <code>2500+ Model Zoo Architectures</code> · <code>392 Test Files</code>
```
with:
```html
<code>339 Lessons</code> · <code>8 Learning Tracks</code> · <code>31 ML Algorithms</code> · <code>8000+ Model Zoo Architectures</code> · <code>393 Test Files</code>
```

- [ ] **Step 2: "What You'll Build" table (lines 36, 41, 51)**

Replace inside the `<sub>` tags:
- Line 36: `736 架构` → `791 架构`
- Line 41: `813 架构` → `814 架构`
- Line 51: `64 架构` → `64 架构` (unchanged — kept for verification)

Only two actual replacements.

- [ ] **Step 3: Vision backbones summary (line 333)**

Replace:
```
<summary><b>支持的 Vision Backbones（208 算法族 / 736 架构 ID）</b></summary>
```
with:
```
<summary><b>支持的 Vision Backbones（208 算法族 / 791 架构 ID）</b></summary>
```

- [ ] **Step 4: Point Cloud backbones summary (line 468)**

Keep line 468 unchanged (`30 算法 / 64 架构 ID` is still accurate for `local_zoo`).

- [ ] **Step 5: Zoo total sentences (lines 753, 1568)**

Line 753 — replace:
```
> 全领域统一模型动物园 — 纯 PyTorch 本地实现，无需下载预训练权重，2500+ 架构 ID 一行切换
```
with:
```
> 全领域统一模型动物园 — 纯 PyTorch 本地实现，无需下载预训练权重，8000+ 架构 ID 一行切换
```

Line 1568 — replace:
```
- **Model Zoo** — 全领域（Vision / NLP / Point Cloud / Multimodal / Generative / Federated）共 2500+ 架构 ID，纯 PyTorch 本地实现，统一接口一行切换
```
with:
```
- **Model Zoo** — 全领域（Vision / NLP / Point Cloud / Multimodal / Generative / Federated）共 8000+ 架构 ID，纯 PyTorch 本地实现，统一接口一行切换
```

- [ ] **Step 6: ASCII tree box (line 1555)**

Replace:
```
              │ 由浅入深       │ 126 pytest  │ 2500+ 架构 ID          │
```
with:
```
              │ 由浅入深       │ 393 pytest  │ 8000+ 架构 ID          │
```

Preserve the column-alignment spacing (the content between `│` must keep the same visual width). If replacement shortens/lengthens, pad with spaces to keep the box pretty. Target visual: each inner cell is 15 chars wide; `393 pytest` = 10 chars + 5 spaces, `8000+ 架构 ID` uses the same width as the old `2500+ 架构 ID`.

- [ ] **Step 7: Verify README grep is clean for known-bad strings**

Run:
```bash
grep -nE "2500\+|126\+ 测试| 76 Lessons|27 ML Algor|392 Test| 126 pytest" README.md
```
Expected output: empty (no matches).

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs(readme): fact sync — refresh stats to 2026-04-18 measurements"
```

---

## Task 3: Edit `docs/index.md`

**Files:**
- Modify: `docs/index.md:24,34,39,44,72,84,184`

- [ ] **Step 1: Replace stat-card values (lines 24, 34, 39, 44)**

Literal replacements (each anchored by the class and preceding line so they are unique):

```
<span class="stat-number">76</span>
<span class="stat-label">Lessons</span>
```
→
```
<span class="stat-number">339</span>
<span class="stat-label">Lessons</span>
```

```
<span class="stat-number">2500+</span>
<span class="stat-label">Model Zoo 架构</span>
```
→
```
<span class="stat-number">8000+</span>
<span class="stat-label">Model Zoo 架构</span>
```

```
<span class="stat-number">27</span>
<span class="stat-label">ML 算法</span>
```
→
```
<span class="stat-number">31</span>
<span class="stat-label">ML 算法</span>
```

```
<span class="stat-number">126</span>
<span class="stat-label">测试文件</span>
```
→
```
<span class="stat-number">393</span>
<span class="stat-label">测试文件</span>
```

`8 Learning Tracks` card stays.

- [ ] **Step 2: Vision / NLP / Point Cloud track cards (lines 72, 84, 108)**

Replace:
- Line 72: `CNN→ViT→检测→分割，736 架构` → `CNN→ViT→检测→分割，791 架构`
- Line 84: `文本分类→Transformer→阅读理解，813 架构` → `文本分类→Transformer→阅读理解，814 架构`
- Line 108: `PointNet→PCT，64 架构` stays.

- [ ] **Step 3: Model Zoo quick link (line 184)**

Replace:
```
2500+ 预定义架构，开箱即用。
```
with:
```
8000+ 预定义架构，开箱即用。
```

- [ ] **Step 4: Verify**

Run:
```bash
grep -nE "2500\+| 76</span>|>27<|>126<|736 架构|813 架构" docs/index.md
```
Expected output: empty.

- [ ] **Step 5: Commit**

```bash
git add docs/index.md
git commit -m "docs(index): fact sync stat-cards and track card numbers"
```

---

## Task 4: Edit `docs/zoo/index.md`

**Files:**
- Modify: `docs/zoo/index.md:7,19,27,71,80`

- [ ] **Step 1: Top banner (line 7)**

Replace:
```
全领域统一模型动物园 --- 纯 PyTorch 本地实现，无需下载预训练权重，**2 500+ 架构 ID** 一行切换。
```
with:
```
全领域统一模型动物园 --- 纯 PyTorch 本地实现，无需下载预训练权重，**8 000+ 架构 ID** 一行切换。
```

- [ ] **Step 2: Per-zoo cards (lines 19, 27)**

Replace:
- Line 19: `**736** Architecture IDs / 208 算法族` → `**791** Architecture IDs / 208 算法族`
- Line 27: `**813** Architecture IDs / 49 算法族` → `**814** Architecture IDs / 49 算法族`
- Line 35 (`**64** Architecture IDs / 30 算法族`): unchanged.

- [ ] **Step 3: Zoo summary table (lines 71, 80)**

Replace:
- Line 71: `| Vision | Backbones | 208 族 / 736 IDs | ... |` → `| Vision | Backbones | 208 族 / 791 IDs | ... |`
- Line 80: `| NLP | Text Encoders | 49 族 / 813 IDs | ... |` → `| NLP | Text Encoders | 49 族 / 814 IDs | ... |`

- [ ] **Step 4: Verify**

Run:
```bash
grep -nE "2 500\+|\*\*736\*\*|\*\*813\*\*|208 族 / 736|49 族 / 813" docs/zoo/index.md
```
Expected output: empty.

- [ ] **Step 5: Commit**

```bash
git add docs/zoo/index.md
git commit -m "docs(zoo/index): fact sync zoo totals and per-zoo counts"
```

---

## Task 5: Edit `docs/zoo/vision-zoo.md`

**Files:**
- Modify: `docs/zoo/vision-zoo.md:7,14`

- [ ] **Step 1: Top banner (line 7)**

Replace:
```
> **208 算法族 / 736 Architecture IDs** --- 覆盖从经典 CNN 到最新 Vision Transformer 的全部视觉主干网络，外加 8 个下游任务子系统。
```
with:
```
> **208 算法族 / 791 Architecture IDs** --- 覆盖从经典 CNN 到最新 Vision Transformer 的全部视觉主干网络，外加 8 个下游任务子系统。
```

- [ ] **Step 2: CLI example comment (line 14)**

Replace:
```
# 列出全部 736 个架构 ID
```
with:
```
# 列出全部 791 个架构 ID
```

- [ ] **Step 3: Verify**

```bash
grep -nE "736|208 算法族 / 736" docs/zoo/vision-zoo.md
```
Expected output: empty.

- [ ] **Step 4: Commit**

```bash
git add docs/zoo/vision-zoo.md
git commit -m "docs(zoo/vision): fact sync 736 → 791"
```

---

## Task 6: Edit `docs/zoo/nlp-zoo.md`

**Files:**
- Modify: `docs/zoo/nlp-zoo.md:7,14`

- [ ] **Step 1: Top banner (line 7)**

Replace:
```
> **49 算法族 / 813 Architecture IDs** --- 覆盖 Transformer、RNN、CNN、MLP 等全部主流文本编码器架构。
```
with:
```
> **49 算法族 / 814 Architecture IDs** --- 覆盖 Transformer、RNN、CNN、MLP 等全部主流文本编码器架构。
```

- [ ] **Step 2: CLI example comment (line 14)**

Replace:
```
# 列出全部 813 个架构 ID
```
with:
```
# 列出全部 814 个架构 ID
```

- [ ] **Step 3: Verify**

```bash
grep -n "813" docs/zoo/nlp-zoo.md
```
Expected output: empty.

- [ ] **Step 4: Commit**

```bash
git add docs/zoo/nlp-zoo.md
git commit -m "docs(zoo/nlp): fact sync 813 → 814"
```

---

## Task 7: Edit `docs/tracks/pointcloud.md`

**Files:**
- Modify: `docs/tracks/pointcloud.md:9,11,52,53`

- [ ] **Step 1: Hero summary (line 9)**

Replace:
```
    **23 个 Lesson**（4 个核心 + 19 个进阶） · 预计 2-3 周 · PointNet、DGCNN、PointNet++ 与 64 架构 Zoo
```
with:
```
    **36 个 Lesson**（4 个核心 + 32 个进阶） · 预计 2-3 周 · PointNet、DGCNN、PointNet++ 与 64 架构 Zoo
```

Rationale: `4 core + 32 advanced = 36`, matching measured total.

- [ ] **Step 2: Intro paragraph (line 11)**

Replace:
```
    Point Cloud 赛道从最经典的 PointNet 出发，逐步引入图卷积（DGCNN）和层级采样（PointNet++），最后通过 30+ Backbone Zoo 统一对比各类 3D 点云架构。赛道还包含自监督学习等进阶内容，共计 23 个 Lesson。
```
with:
```
    Point Cloud 赛道从最经典的 PointNet 出发，逐步引入图卷积（DGCNN）和层级采样（PointNet++），最后通过 30+ Backbone Zoo 统一对比各类 3D 点云架构。赛道还包含自监督学习等进阶内容，共计 36 个 Lesson。
```

- [ ] **Step 3: Admonition title (line 52)**

Replace:
```
!!! note "23 个 Lesson 总计"
```
with:
```
!!! note "36 个 Lesson 总计"
```

- [ ] **Step 4: Admonition body (line 53)**

Replace:
```
    除上述 4 个核心 Lesson 外，Point Cloud 赛道还包含 **19 个进阶 Lesson**，涵盖自监督点云预训练（15 种方法）、部件分割、场景分割、点云重建等主题，共计 23 个 Lesson。
```
with:
```
    除上述 4 个核心 Lesson 外，Point Cloud 赛道还包含 **32 个进阶 Lesson**，涵盖自监督点云预训练（15 种方法）、部件分割、场景分割、点云重建等主题，共计 36 个 Lesson。
```

- [ ] **Step 5: Leave line 113 unchanged**

`!!! note "64 架构可供切换"` — kept (backbone count unchanged).

- [ ] **Step 6: Verify**

```bash
grep -nE "23 个 Lesson|19 个进阶" docs/tracks/pointcloud.md
```
Expected output: empty.

- [ ] **Step 7: Commit**

```bash
git add docs/tracks/pointcloud.md
git commit -m "docs(tracks/pointcloud): fact sync lessons 23 → 36"
```

---

## Task 8: Edit `docs/tracks/vision.md`

**Files:**
- Modify: `docs/tracks/vision.md:123`

- [ ] **Step 1: Admonition title**

Replace:
```
!!! note "736 架构可供切换"
```
with:
```
!!! note "791 架构可供切换"
```

- [ ] **Step 2: Verify**

```bash
grep -n "736 架构" docs/tracks/vision.md
```
Expected output: empty.

- [ ] **Step 3: Commit**

```bash
git add docs/tracks/vision.md
git commit -m "docs(tracks/vision): fact sync 736 → 791"
```

---

## Task 9: Edit `docs/tracks/nlp.md`

**Files:**
- Modify: `docs/tracks/nlp.md:134`

- [ ] **Step 1: Admonition title**

Replace:
```
!!! note "813 架构可供探索"
```
with:
```
!!! note "814 架构可供探索"
```

- [ ] **Step 2: Verify**

```bash
grep -n "813 架构" docs/tracks/nlp.md
```
Expected output: empty.

- [ ] **Step 3: Commit**

```bash
git add docs/tracks/nlp.md
git commit -m "docs(tracks/nlp): fact sync 813 → 814"
```

---

## Task 10: Edit `docs/tracks/index.md`

**Files:**
- Modify: `docs/tracks/index.md:66`

- [ ] **Step 1: Pointcloud row**

Replace:
```
| [**Point Cloud** 点云](pointcloud.md) | 23 | PointNet, DGCNN, PointNet++, 64 架构 Zoo | Vision + 3D 几何直觉 |
```
with:
```
| [**Point Cloud** 点云](pointcloud.md) | 36 | PointNet, DGCNN, PointNet++, 64 架构 Zoo | Vision + 3D 几何直觉 |
```

- [ ] **Step 2: Check other track lesson counts in the table**

Run:
```bash
grep -nE "^\| \[\*\*" docs/tracks/index.md
```
For each row, cross-check the Lesson column against measured values:

| Track | Measured | Fix if different |
|---|---|---|
| Foundations | 2 | — |
| Vision | 89 | — |
| NLP | 49 | — |
| GNN | 11 | — |
| Point Cloud | 36 | already done above |
| Generative | 51 | — |
| LLM | 43 | — |
| Multimodal | 58 | — |

Update any row whose Lesson count is numerically different from the measured value. Change ONLY the number column.

- [ ] **Step 3: Verify**

```bash
grep -nE "\| 23 \|" docs/tracks/index.md
```
Expected output: empty.

- [ ] **Step 4: Commit**

```bash
git add docs/tracks/index.md
git commit -m "docs(tracks/index): fact sync per-track lesson counts"
```

---

## Task 11: Edit `docs/developer/structure.md`

**Files:**
- Modify: `docs/developer/structure.md:22`

- [ ] **Step 1: Replace**

```
├── tests/              # 126+ 测试文件
```
with:
```
├── tests/              # 393 测试文件
```

Preserve the column alignment (the `#` column must stay in the same position; `393 测试文件` is shorter than `126+ 测试文件` by 1 char — acceptable, the tree is not strictly aligned).

- [ ] **Step 2: Verify**

```bash
grep -n "126+ 测试" docs/developer/structure.md
```
Expected output: empty.

- [ ] **Step 3: Commit**

```bash
git add docs/developer/structure.md
git commit -m "docs(developer/structure): fact sync test count 126+ → 393"
```

---

## Task 12: Edit `docs/changelog.md`

**Files:**
- Modify: `docs/changelog.md` (top insert + line 105)

- [ ] **Step 1: Prepend a new section at the top of the changelog**

Open `docs/changelog.md`. The file currently has a `# 更新日志` or `# Changelog` heading at the top followed by dated sections. Find the line right after the H1 heading and before the first dated section, and insert:

```markdown

## 2026-04-18 · 文档事实同步

- 刷新 README 与 docs/ 全站统计数字至实测值：Lessons 339 / 测试 393 / ML 算法 31 / Zoo 架构 8545。
- 校正 Vision backbone 791 / NLP backbone 814 / Pointcloud 赛道 36 Lessons。
- 详见 `docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md`。

```

(leading and trailing blank lines included so the section is well-separated).

- [ ] **Step 2: Leave line 105 unchanged**

Historical entry `测试覆盖率提升至 126+ 测试文件` — do NOT rewrite history. The correction is captured in the new top section.

- [ ] **Step 3: Verify**

```bash
grep -n "2026-04-18 · 文档事实同步" docs/changelog.md
```
Expected output: 1 match.

- [ ] **Step 4: Commit**

```bash
git add docs/changelog.md
git commit -m "docs(changelog): record 2026-04-18 fact sync entry"
```

---

## Task 13: Project-wide verification grep

**Files:**
- Modify: none

- [ ] **Step 1: Run the must-be-zero grep suite**

Run:
```bash
grep -rnE "2500\+|2 500\+" README.md docs/*.md docs/tracks/*.md docs/zoo/*.md docs/developer/*.md docs/getting-started/*.md
```
Expected: empty.

```bash
grep -rnE " 76 Lessons|27 ML Algor|392 Test|126\+ 测试" README.md docs/*.md docs/tracks/*.md docs/zoo/*.md docs/developer/*.md docs/getting-started/*.md
```
Expected: empty.

```bash
grep -rnE "736 架构|813 架构|23 个 Lesson|19 个进阶" docs/tracks docs/zoo
```
Expected: empty.

```bash
grep -rnE "208 算法族 / 736|49 算法族 / 813" docs/zoo
```
Expected: empty.

- [ ] **Step 2: Confirm `docs/plans/` untouched**

Run:
```bash
git diff --name-only HEAD~12..HEAD -- docs/plans/
```
Expected: empty.

- [ ] **Step 3: Run mkdocs build (optional, if env available)**

Run:
```bash
pip install -q -r requirements-docs.txt 2>/dev/null
mkdocs build --strict 2>&1 | tail -20
```
Expected: no new WARN/ERROR vs. baseline. If the env lacks the deps, skip and note in the final report.

- [ ] **Step 4: If any grep returns non-empty, add a fix commit**

Do NOT amend previous commits. Add a new commit with message `docs: fact sync — cleanup missed occurrences` containing only the additional edits.

---

## Task 14: Final report

**Files:**
- Modify: none

- [ ] **Step 1: Produce a summary for the user**

Output the following:

```
Docs fact sync complete.

Files modified: 11 (README.md + 10 docs/*.md)
Commits: 12 (task 2 — task 12, plus optional task 13 cleanup)

Verification:
- grep suite: all 0 matches (§Task 13)
- mkdocs build: <pass / skipped — reason>

Key deltas:
- Lessons: 76 → 339 (docs/index.md); pointcloud track 23 → 36
- Tests: 126+ / 392 → 393
- ML algorithms: 27 → 31
- Zoo total: 2500+ → 8000+
- Vision backbone: 736 → 791
- NLP backbone: 814
- Pointcloud backbone: 64 (unchanged)

Spec: docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md
Plan: docs/superpowers/plans/2026-04-18-docs-fact-sync-plan.md
```

- [ ] **Step 2: Do NOT push**

Leave commits local. The user decides when to push.
