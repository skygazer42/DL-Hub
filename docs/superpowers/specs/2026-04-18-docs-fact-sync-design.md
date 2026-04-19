# DL-Hub Docs 事实同步 & 统计刷新 · 设计文档

- **日期**: 2026-04-18
- **作者**: jhlu
- **范围**: `README.md` + `docs/` 下的 Markdown 文件（不含 `docs/plans/`）
- **策略**: A — 最小字面修正（不改范围解释，不新增内容，不引入脚本/CI）

## 1. 背景

`README.md` 的头部徽标与 `docs/index.md`、`docs/zoo/*.md`、`docs/tracks/*.md`、`docs/developer/structure.md`、`docs/changelog.md` 中的若干计数与现状严重脱节：

- `docs/index.md` 仍写 `76 Lessons`、`126 测试文件`；
- 所有 zoo 类文档把总架构数写为 `2500+`；
- Vision / NLP / Pointcloud 的 backbone 计数分别停留在 `736 / 813 / 64`；
- `docs/tracks/pointcloud.md` 与 `docs/tracks/index.md` 的 Pointcloud lesson 数停留在 `23`。

本次任务在不改变文档语义、范围、结构的前提下，把这些数字同步到 2026-04-18 真实测得的值。

## 2. 测量方法（可重现）

以下命令即本次测量的权威来源。后续任何同类刷新均可复用。

### 2.1 Lessons 总数（按 track）

```bash
for d in tracks/*/; do
  name=$(basename "$d")
  [ "$name" = "__pycache__" ] && continue
  count=$(find "$d" -maxdepth 1 -mindepth 1 -type d -name "lesson_*" | wc -l)
  echo "$name: $count"
done
find tracks -maxdepth 2 -mindepth 2 -type d -name "lesson_*" | wc -l
```

### 2.2 测试文件数

```bash
find tests -name "test_*.py" -type f | wc -l
```

### 2.3 ML 算法数

```bash
ls ml_algorithms/python/*.py | grep -v __init__ | wc -l
```

### 2.4 Zoo 架构数

```python
# python -c 调用每个 *_zoo.py 的 list_local_arches()
from pathlib import Path
from collections import defaultdict
import importlib

bucket = defaultdict(int)
for p in Path('dlhub').rglob('*_zoo.py'):
    mod = str(p.with_suffix('')).replace('/', '.')
    m = importlib.import_module(mod)
    for fn_name in ('list_local_arches', 'list_local_archs'):
        if hasattr(m, fn_name):
            top = p.parts[1] if len(p.parts) >= 2 else 'root'
            bucket[top] += len(getattr(m, fn_name)())
            break
print(sum(bucket.values()), dict(bucket))
```

## 3. 真实值（2026-04-18）

| 维度 | 旧值（多处） | 新值 | 测量依据 |
|---|---|---|---|
| Lessons 总数 | 76 / 339 | **339** | §2.1 |
| Lessons · foundations | — | **2** | §2.1 |
| Lessons · generative | — | **51** | §2.1 |
| Lessons · gnn | — | **11** | §2.1 |
| Lessons · llm | — | **43** | §2.1 |
| Lessons · multimodal | — | **58** | §2.1 |
| Lessons · nlp | — | **49** | §2.1 |
| Lessons · pointcloud | 23 | **36** | §2.1 |
| Lessons · vision | — | **89** | §2.1 |
| Test 文件 | 126+ / 392 | **393** | §2.2 |
| ML 算法文件 | 27 | **31** | §2.3 |
| Zoo 架构总数 | 2500+ | **8000+**（精确 8545） | §2.4 |
| Vision backbone (`dlhub/vision/local_zoo.py`) | 736 | **791** | §2.4 |
| NLP backbone (`dlhub/nlp/local_zoo.py`) | 813 | **814** | §2.4 |
| Pointcloud backbone (`dlhub/pointcloud/local_zoo.py`) | 64 | **64**（保留） | §2.4 |

## 4. 修改清单

每处只替换数字，不改写句式、不新增段落。

### 4.1 `README.md`

- 第 18 行（或 `339 Lessons · 27 ML Algorithms · 2500+ Model Zoo Architectures · 392 Test Files`）→
  `339 Lessons · 31 ML Algorithms · 8000+ Model Zoo Architectures · 393 Test Files`。
- 其余徽标/段落若再次出现同类数字，同步替换（用 4.8 的 grep 清单兜底）。

### 4.2 `docs/index.md`

- `stat-card` × 5：`76 Lessons` → `339`；`2500+` → `8000+`；`27` → `31`；`126` → `393`；`8 Learning Tracks` 保持。
- Vision/NLP/Point Cloud 三张学习赛道卡片的内文 `736 架构` → `791`；`813 架构` → `814`；`64 架构` 保持。

### 4.3 `docs/zoo/index.md`

- 顶部 `2 500+ 架构 ID` → `8 000+ 架构 ID`。
- 三个分卡：`736` → `791`；`813` → `814`；`64` 保持。

### 4.4 `docs/zoo/vision-zoo.md`

- 摘要 `208 算法族 / 736 Architecture IDs` → `208 算法族 / 791 Architecture IDs`。
- CLI 示例注释 `# 列出全部 736 个架构 ID` → `791`。

### 4.5 `docs/zoo/nlp-zoo.md`

- `49 算法族 / 813 Architecture IDs` → `49 算法族 / 814 Architecture IDs`。
- 其它 `813` 同步。

### 4.6 `docs/zoo/pointcloud-zoo.md`

- 保持 `30 算法族 / 64 Architecture IDs`；保持 `4 个 3D 下游任务子系统` 不动（本次不重计子系统数）。

### 4.7 `docs/zoo/generative-zoo.md`、`federated-zoo.md`、`vlm-zoo.md`

- 默认**不改动**。仅当顶部摘要某条数字能在 1 分钟内用已有测量命令（§2）验证失真、且替换文本即可修正时，才同步；否则留到下次改造批次。
- 本次不为这三份文档新增章节或子系统列表。

### 4.8 `docs/tracks/`

- `tracks/pointcloud.md`：`23 个 Lesson`（包括 `(4 个核心 + 19 个进阶)` 的括注）→ `36 个 Lesson`；`64 架构` 保持。
- `tracks/vision.md`：`736 架构` → `791 架构`（含 admonition）。
- `tracks/nlp.md`：`813 架构` → `814 架构`。
- `tracks/index.md`：Pointcloud 行 `23` → `36`；若其它 track 行有老数字一并修正。

### 4.9 `docs/developer/structure.md`

- `126+ 测试文件` → `393 测试文件`。

### 4.10 `docs/changelog.md`

- 追加一条顶部变更说明：
  ```markdown
  ## 2026-04-18 · docs 事实同步
  - 刷新统计数字至实测值：Lessons 339 / 测试 393 / ML 算法 31 / Zoo 架构 8005
  - 同步 README 与 docs 下 stat-card、zoo 摘要、tracks 架构计数
  - 来源：docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md
  ```
- 不重写历史条目。

## 5. 不做

- 不修改 `docs/plans/` 下任何文件（视为历史规划档案）。
- 不新增脚本、不接入 CI、不引入 mkdocs include 机制。
- 不重写章节结构、不合并 `CONVENTIONS.md`/`INSTALL.md`/`RUNNING.md`/`STRUCTURE.md`/`STYLEGUIDE.md`/`ROADMAP.md`。
- 不补新增的 vision zoo 子模块文档（anchor_free_detection、deepfake_detection、face/hand/finger 系列等）。

## 6. 验收

1. grep 兜底（必须全部命中 0）：

   ```bash
   grep -rnE "2500\+|126\+| 76 Lessons|27 ML Algor|392 Test|2 500\+| 126\+" README.md docs/*.md docs/**/*.md
   grep -rnE "736 架构|813 架构|23 个 Lesson" docs/tracks docs/zoo
   grep -rn "736 个架构 ID\|208 算法族 / 736\|49 算法族 / 813" docs/zoo
   ```

2. 手工目视确认：`docs/index.md` 顶部 5 张 stat-card 数字正确；`docs/zoo/index.md` 三张分卡正确；`docs/tracks/pointcloud.md` 首页数字与 `tracks/index.md` 一致。

3. 如环境可用，运行 `mkdocs build --strict`（`requirements-docs.txt`），不新增 WARN/ERROR。

4. git diff 行数应落在 ~60 行上下，且绝大多数是数字替换，如偏离过大说明本次范围外改动，退回。

## 7. 风险与回滚

- **风险 1**：某些 "算法族" 数字（208 / 49 / 30 / 36）含义为 backbone family 的纯计数，无自动脚本校验，仅相对稳定。若实测 backbone 族数已变，按 backbone 实测值改；本次若未能快速测得，保持原值。
- **风险 2**：`changelog.md` 的追加条目若与团队已有撰写风格不符，可在评审阶段调整句式。
- **回滚**：所有改动在一个 commit，`git revert` 即可恢复。

## 8. 交付物

1. 本 spec：`docs/superpowers/specs/2026-04-18-docs-fact-sync-design.md`
2. 后续由 writing-plans 产出的执行计划：`docs/superpowers/plans/2026-04-18-docs-fact-sync-plan.md`
3. 最终 PR / commit：单次提交，消息格式 `docs: fact sync — refresh stats to 2026-04-18 measurements`
