# 平台运行门禁

`scripts/platform_runtime_check.py` 用真实计算和真实文件系统操作验证平台边界。它不会因为
主机名、环境变量或模拟对象看起来像 CUDA、MPS、Windows 或 NFS 就报告通过。

## 门禁实际执行什么

每次成功运行都会完成以下检查：

- 在解析出的 PyTorch 设备上执行矩阵乘法和 12 步小型 SGD 训练，检查有限 loss、参数更新和
  loss 下降；
- 从该设备保存 state-dict checkpoint，再以 `map_location="cpu"` 加载到 CPU model 和
  optimizer；加载始终使用 `weights_only=True`；
- 让故意在写入中途失败的 writer 证明旧文件仍完整，且 sibling 临时文件已清理；
- 多进程竞争替换同一 JSON，同时由读者持续解析，拒绝部分内容；
- 多进程追加 JSONL，验证每条记录完整、无丢失、无重复；
- 清理运行工作区；POSIX 默认 30 秒超时，超时会明确失败而不是报告成功。

成功和失败结果都可用 `--output` 原子写为 JSON。结果包含 `ok`、平台/文件系统/设备证据、
Git HEAD、dirty 状态、门禁脚本及关键 `dlhub` 文件的 SHA-256；GitHub Actions 还会记录
`GITHUB_SHA` 和 `GITHUB_RUN_ID`。因此 dirty worktree 的一次实测也能追溯到确切源码字节。

## 本地运行

先按[安装指南](../getting-started/installation.md)安装适配本机的 PyTorch，再从仓库根目录运行：

```bash
# 可移植 CPU + 当前文件系统
python scripts/platform_runtime_check.py \
  --device cpu \
  --require-device cpu \
  --output outputs/runtime-audit/platform-runtime-cpu.json

# Linux CUDA；与其他训练任务共用锁时
flock outputs/runtime-audit/cuda.lock \
  python scripts/platform_runtime_check.py \
    --device cuda:0 \
    --require cuda \
    --output outputs/runtime-audit/platform-runtime-cuda.json

# 必须指向已经挂载的真实 NFS；结果写到 NFS 外
python scripts/platform_runtime_check.py \
  --device cpu \
  --require nfs \
  --work-dir /path/to/nfs-mount \
  --output /path/outside-nfs/platform-runtime-nfs.json
```

统一的 `--require` 可重复使用：

| 参数 | 硬性成功条件 |
|------|--------------|
| `--require cuda` | `resolve_device` 得到 CUDA，张量、训练和 checkpoint 确实在 CUDA 源设备执行 |
| `--require mps` | PyTorch 报告 MPS 可用，且同一组设备操作确实在 `mps` 执行 |
| `--require windows` | `platform.system()` 为 Windows，且 Win32 `GetVolumeInformationW` 返回真实卷文件系统 |
| `--require nfs` | Linux mountinfo 中目标路径的 `st_dev` 对应 `nfs` 或 `nfs4`，不是名称相似的本地目录 |

`--require-device` 和 `--require-filesystem` 保留为细粒度兼容入口。要求冲突、后端不可用、
文件系统证据不匹配或任何运行检查失败时，进程退出码均非零，`--output` 中为 `ok: false`。

### NFS 清理边界

硬挂载 NFS 可能处于 server grace/reclaim 窗口，客户端 `rmdir` 也可能长时间等待。由外部临时
mount harness 负责生命周期时，可加 `--keep-workspace`，把结果写到 NFS 外；只有在 Python
成功退出、客户端卸载后，harness 才从 export 端删除该次唯一工作目录。JSON 会明确标为
`workspace_cleanup.status: retained`，外部清理必须另行验证，不能把 retained 当成已清理。

门禁对子进程目前没有独立超时；本地自动化应再提供进程级截止时间。GitHub workflow 有
30 分钟 job timeout，脚本自己的 POSIX 工作区删除默认有 30 秒截止时间。

不要对不可信仓库内容开放 privileged NFS 容器。项目的常规 CI 不启动 NFS server；本机
loopback NFS 只用于受控、一次性验收。

## Windows 与 MPS runner

`.github/workflows/platform-runtime.yml` 是独立 workflow，不是尚未实测就强加给每个 PR 的
门禁。它提供两个受控入口：

- 合入默认分支后，可用 `workflow_dispatch` 选择 `all`、`windows` 或 `mps`；GitHub 官方要求
  手动触发的 workflow 文件先存在于默认分支；
- 首次合入前，可推送严格限定为 `platform-runtime-verify/**` 的临时验证分支；该入口固定同时
  运行 Windows 与 MPS，不接受普通分支或 PR 触发。

两个入口运行相同的原生 job：

- `windows-2025` 使用 Python 3.12，运行平台测试和 CPU 门禁，再真实运行 foundations 线性
  回归短课，解析 config/metrics 并检查 checkpoint/log；无论成功失败都上传 JSON 和课程产物；
- `macos-15` 使用 GitHub 标准 Apple Silicon runner，显式要求 `mps`，上传成功或失败 JSON。

原生证据矩阵固定 Python 3.12 和 PyTorch 2.13.0，避免归档结果随依赖解析漂移；这不改变项目
公开的 `torch>=2.0` 兼容范围。`macos-14` 虚拟化 runner 虽会返回 MPS available，却不能完成
首次张量分配；项目根据 [runner-images #9918](https://github.com/actions/runner-images/issues/9918)
的原生限制和实际失败证据改用 `macos-15`，没有降级到 CPU 或跳过 MPS。

所用 runner 标签与架构来自 [GitHub-hosted runners 官方表](https://docs.github.com/en/actions/reference/runners/github-hosted-runners/)。
PyTorch 要求用 `torch.backends.mps.is_available()` 并把 tensor/module 真正移到 `mps`；参见
[PyTorch MPS 说明](https://docs.pytorch.org/docs/stable/notes/mps.html) 和
[Apple 的 PyTorch Metal 指南](https://developer.apple.com/metal/pytorch/)。workflow 的 MPS
步骤执行的正是这条实际计算边界。

无论使用手动入口还是临时验证分支，只有 Actions 日志和上传 JSON 成功后才构成该平台的
实测证据。workflow 文件存在、runner 标签官方可用或本地 mock 通过，都不等于 Windows/MPS
已实测。
Windows job 只运行一门 foundations 短课；它不等于 339 个 lesson 的 full run。下节记录的
339 默认预算结果来自 **Linux/CUDA**，不能外推为 Windows 或 MPS 课程通过。

## 2026-08-30 本机证据

| 边界 | 结果 | 实测证据 |
|------|------|----------|
| Linux/ext4 + CUDA | **通过** | RTX 3070 Ti（8,218,804,224 bytes，compute capability 8.6），PyTorch 2.4.0+cu121 / CUDA 12.1；matmul、训练、CUDA 保存后 CPU 回读 checkpoint 全过，4.948 秒 |
| loopback NFSv4.2 | **通过** | 宿主 ext4 目录经一次性 privileged Docker NFSD 导出，客户端 `findmnt` 为 `127.0.0.1:/` / `nfs4` / `vers=4.2`，设备 `0:357`；32 条并发 JSONL、12 次竞争 replace、故障 writer、CPU 训练和 checkpoint 全过，117.928 秒（包含 NFSv4 grace 等待） |
| Windows | **通过** | GitHub `windows-2025` / X64，Windows Server 2025、NTFS、Python 3.12.10、PyTorch 2.13.0+cpu；平台门禁和一门真实离线 lesson 及其四类标准产物全部通过 |
| Apple MPS | **通过** | GitHub `macos-15` / ARM64，Darwin 24.6、Python 3.12.10、PyTorch 2.13.0；MPS build/availability、真实张量、训练及 MPS checkpoint 到 CPU 回读全部通过 |

CUDA 与 NFS 成功 JSON 绑定到同一门禁脚本 SHA-256
`4bde73bc1373354214864695784812005d784c9acaabdfab1d465665ec246890`。NFS 容器、mount、
retained workspace、fixture 和 client 目录均已删除；本地忽略目录只保留成功 JSON 与不含
临时绝对路径的摘要。这些本地证据不随 Git 提交分发，应以 JSON 中的源码哈希和实际执行
日志复核。

### 原生 Windows 与 MPS 的最终证据

最终成功运行是 [GitHub Actions #33304800747](https://github.com/skygazer42/DL-Hub/actions/runs/33304800747)，
绑定干净提交 `1928efbf98b33bc87fea7962539525f57a349d25`。Windows job
`99239323612` 和 MPS job `99239323488` 均为 success；下载后的日志没有 `##[error]` 或非零退出。

| 项目 | Windows | Apple MPS |
|------|---------|-----------|
| runner 镜像 | `win25-vs2026` `20260824.214.3` / X64 | `macos15` `20260727.0256.1` / ARM64 |
| 平台运行 | NTFS 由 Win32 API 实测；CPU matmul、12 步训练、checkpoint 通过，5.459 秒 | `mps_built=true`、`mps_available=true`；MPS matmul、12 步训练通过，loss `1.430945 -> 0.362996`，5.590 秒 |
| 文件边界 | 12 次竞争 replace、故障 writer 保留旧文件、32 条跨进程 JSONL 无丢失/重复 | 同一组 replace、故障 writer 和 32 条跨进程 JSONL 全通过 |
| checkpoint | 从 CPU 序列化，以 `weights_only=True` 在 CPU 恢复 model/optimizer | 从 MPS 序列化，以 `weights_only=True` 在 CPU 恢复 model/optimizer |
| 额外课程证据 | foundations lesson 02：1 epoch、2 train batches、1 eval batch；config、1 条 finite metrics、checkpoint、日志均验真 | 不把 Windows 短课或 Linux 339 课程外推成 MPS 课程全量通过 |

MPS JSON、Windows JSON 的 SHA-256 分别为
`db67cf0bb8bb2bf9f9458dd2adf986e8ddf8ba7ec9cc4a36479e1cdf781a1ee5` 和
`58d2d0f118e7d4d2817c940ffccc5f099862d4311f8c758951dae1d230eeb550`；下载归档分别为
`e32f5d6061ab78db044288fb1bf685eee7314b28639c7f5ae2667b64af7ffbef` 和
`d2650b44e9bec0036d112bb87035ba0bf9fff65b59c5cc268aec91468404bde6`。两个 JSON 的五项
源码 SHA-256 完全一致，并与当前工作树字节一致；Windows lesson checkpoint 的 SHA-256 为
`a4b81de6b9d67b0ed4bfb226f9a204bfcca3f0268d28f9c2dbd27875288db77a`。

原生验证没有隐藏失败尝试：前几轮依次暴露并修复了 Windows 缺少 `os.makedev`、Linux
mountinfo 被 Windows 路径语义改写、并发 `os.replace` sharing violation、跨进程 JSONL
丢记录和原子读的短暂 sharing window；MPS 侧证明 `macos-14` 在 PyTorch 2.12/2.13 及禁用
allocator watermark 时仍无法分配，迁到 `macos-15` 后连续通过。功能双绿的
[run #33304657097](https://github.com/skygazer42/DL-Hub/actions/runs/33304657097) 还发现 Windows
checkout 的 CRLF 导致源码证据哈希不同；最终轮加入 LF 属性后，跨平台哈希也完成收口。

## 339 个 lesson 的 Linux/CUDA 默认预算验收

最终报告位于忽略目录
`outputs/runtime-audit/runs/full-cuda-defaults-339-final-20260830/report.json`，报告 SHA-256 为
`42a2387b8ef0fa63556da22a1e7968f8fe8d44af097245cbc782eb00f5a1ed88`。这是一次串行、
离线优先、默认课程预算的 Linux/CUDA 实测快照：

| 项目 | 最终结果 |
|------|----------|
| 课程终态 | **339/339 passed**：338 个 `train` 入口、1 个 `run` 入口 |
| 训练预算 | 默认参数估算合计 **71,360 batches**；最终命令没有 `--max-train-batches` 或 `--max-eval-batches` 截断 |
| 墙钟时间 | **3215.479 秒** |
| 标准训练产物 | **338/338** 均有通过校验的 `config.json`、`metrics.jsonl`、`logs/train.log` 和可用 `weights_only=True` 安全加载的 checkpoint |
| 指标 | 共 **2,633** 条 JSONL metric records，JSON 均可解析且所有数值有限 |
| CUDA 分配证据 | 338 个训练入口中 **337** 个记录到正的 child `torch.cuda.max_memory_allocated` peak |
| 唯一零 peak | `nlp/lesson_12_compact_in_context_text_classification` 明确产出 `model_free: true` checkpoint，因此零 CUDA tensor peak 符合实现 |
| model-free 对照 | `gnn/lesson_05_label_propagation_cora` 同样是 model-free checkpoint，但运行中仍记录 **823,296 bytes** 正 CUDA peak；`model_free` 不等于必然零分配 |
| 日志与清理 | 运行日志中 `Traceback`、`AF_UNIX`、nested-tensor 相关匹配均为 0；临时链接为 0，验收时 CUDA lock 可立即获取 |

报告绑定的源码树 SHA-256 为
`956e8c029a23868c66b3efdc025323797a127760b1656a495e8b46c672059ed3`，课程 inventory
SHA-256 为 `502451551af1098a1e1f798b5381c977f5dd87cbc38e08ec729dc7cced22fd6e`，
Git diff SHA-256 为 `022d15cce27b339db8eb2dd817147744639f7666256ce2e67f100600f4fb8f99`。
这些哈希用于把 dirty worktree 上的结果绑定到实际执行源码和课程清单。

### 结论边界

这份 `339/339` 证明的是仓库当前 **离线路径、默认 lesson 预算、CUDA 可执行性和标准产物
契约**。它不证明真实数据集训练已经执行，不证明论文 benchmark、论文指标或模型保真度，
也不应被写成跨 Windows/MPS 的成功结论。

运行器设置离线环境变量，并通过 Python `sitecustomize` 阻止 AF_INET/AF_INET6 socket；
这覆盖进入同一 Python 启动链的进程，但外部二进制子进程并未放进系统级网络沙箱。因此
“运行日志没有网络阻断错误”不能升级为绝对断网证明。
