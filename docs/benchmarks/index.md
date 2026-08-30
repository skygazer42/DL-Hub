---
icon: material/file-certificate-outline
---

# 数据、运行与 Benchmark 证据

这页把三个容易混淆的问题拆开：代码能否执行、是否使用真实数据、是否达到论文 benchmark。
结论先说清楚：仓库已有一次 **339/339 默认课程预算的离线 CUDA 运行证据**，但那次运行中
7 个可切换数据集的课程明确选择了 `fake`；仓库目前有真实 MNIST 的可复现执行方案，
**尚没有把这 21 次计划运行写成已完成报告，也没有论文 benchmark 证据**。

## 三份机器可检验证据

| 文件 | 覆盖范围 | 可以证明 | 不能证明 |
|------|----------|----------|----------|
| [`lesson-evidence.json`](lesson-evidence.json) | 339 个 lesson | 每门课的数据路由、README benchmark 分级和允许的声明层级 | 训练已经运行 |
| [`profiles.json`](profiles.json) | 7 个 MNIST lesson × 3 seeds | 真实数据、默认预算、命令、指标和报告契约已经固定 | 21 次运行已经完成 |
| [`runtime-attestation.json`](runtime-attestation.json) | 历史 Linux/CUDA 全量报告 | 绑定报告哈希的 339/339 离线默认预算执行结果 | 真实数据、论文指标或论文实现保真度 |

日常门禁会重新从源码生成前两份文件，并校验第三份证明的内容哈希和声明边界：

```bash
make evidence
# 等价于
python scripts/benchmark_profiles.py --check
```

`make verify` 已包含这个检查。新增 lesson、修改数据 CLI、README benchmark 表述或 MNIST
实现后，使用 `python scripts/benchmark_profiles.py --write` 更新证据文件；重写历史运行证明
还要求本地存在对应的原始报告。

## 已观测：339 门离线默认预算运行

历史原始报告位于忽略目录
`outputs/runtime-audit/runs/full-cuda-defaults-339-final-20260830/report.json`。签入仓库的是其精简
证明，不是 3.1 MiB 的运行日志副本：

| 项目 | 已观测值 |
|------|----------|
| 原始报告 SHA-256 | `42a2387b8ef0fa63556da22a1e7968f8fe8d44af097245cbc782eb00f5a1ed88` |
| 终态 | 339/339 passed，全部一次尝试 |
| 数据选择 | 7 个命令显式使用 `--dataset fake`；其余 332 个使用入口内置离线路径 |
| 截断 | 0 个命令含 `--max-*` |
| 标准训练产物 | 338/338 通过校验 |
| 指标 | 2,633 条有限 JSONL 记录 |
| CUDA peak | 338 个训练入口中 337 个记录到正值；另 1 个是声明为 `model_free` 的课程 |

该证明记录执行设备、源码树哈希、Git diff 哈希和课程 inventory 哈希。它是历史源码快照的
运行证据；后续源码修改不会被倒推成“同一份结果仍然代表当前代码”。完整运行边界见
[平台运行门禁](../developer/platform-runtime.md)。

## 已定义、未执行：`mnist-real-v1`

真实数据 profile 固定以下规则：

- 数据：`torchvision.datasets.MNIST` 的公开真实数据；完成报告必须记录本地数据文件 SHA-256。
- 矩阵：7 个 lesson × seeds `41, 42, 43`，共 21 次。
- 预算：保留每个入口当前的 epochs、batch size、算法步数和数据规模默认值。
- 唯一允许覆盖：`--dataset mnist`、`--seed`、`--device`、`--run-name`。
- 禁止：`--max-*`，以及覆盖 epochs、batch size、steps 或 `--num-samples`。
- 声明：完成报告只证明这些紧凑课程在真实 MNIST 上执行并产出有限指标，不自动成为论文复现。

查看精确课程、源码哈希和默认预算：

```bash
python scripts/benchmark_profiles.py --list
python scripts/benchmark_profiles.py --commands mnist-real-v1 --device cuda
```

`--commands` 输出完整的 21 条命令，但不会执行它们。这样“生成计划”和“已经运行”不会被
一个 CLI 动作混在一起。执行环境应保存命令、数据文件哈希、源码快照、环境、逐 seed 产物和
逐 epoch 指标，并用以下命令验证最终报告：

```bash
python scripts/benchmark_profiles.py \
  --validate-report outputs/benchmark-runs/<run-id>/report.json
```

验证器要求 21 个 `(lesson, seed)` 组合无遗漏、所有状态通过、标准产物带 SHA-256、末条所需
指标有限，并拒绝 `--max-*`、预算覆盖和 `paper_benchmark_evidence: true`。

## Benchmark 分级

339 个 README 的当前静态结果是：335 个仅有执行层级表述，3 个只有 `[0, 1]` 一类接受范围，
1 个明确是本地离线 benchmark；需要人工复核的 benchmark 声明为 0。这里的
`local-offline-benchmark` 仍然不是论文 benchmark。

`smoke` 不会从仓库中消失：它是快速验证链路的测试术语。它可以证明“最短路径能运行”，
不能证明真实数据、论文机制或论文指标。证据门禁的目的正是防止 smoke、注册数量或类名被
包装成更强结论。
