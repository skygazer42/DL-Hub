# Lesson 14：合成视频多目标跟踪（MOT Basics）

这节课用一个 **完全离线** 的 toy 视频数据集，跑通 MOT 最小训练闭环：

- 数据：多目标矩形在短视频里做线性运动（可控速度、数量、类别）
- 模型：复用本仓库 `dlhub.vision.mot_zoo`（默认 `mot2d:sort_tiny`）
- 训练：box 回归 + existence score + 类别预测
- 指标：presence-acc、mean IoU（按存在目标统计）

> 说明：这是教学用 MOT skeleton 训练，不包含 Hungarian matching、IDF1/HOTA、轨迹后处理等工程化模块。

## 先选模型（80 算法族）

先看有哪些 family：

```bash
python -m tracks.vision.lesson_14_video_mot_basics.train --list-arch-families
```

看某个 family 的变体（tiny/small/base）：

```bash
python -m tracks.vision.lesson_14_video_mot_basics.train --list-arch --arch-family sort --list-sort alpha
```

关键字筛选：

```bash
python -m tracks.vision.lesson_14_video_mot_basics.train --list-arch --arch-match track --list-limit 30
```

按场景推荐（便于先拍板）：

```bash
python scripts/mot_zoo.py --recommend realtime --top-k 8 --variant tiny
python scripts/mot_zoo.py --recommend occlusion --top-k 8 --variant tiny
python scripts/mot_zoo.py --recommend long_horizon --top-k 8 --variant tiny
python scripts/mot_zoo.py --recommend occlusion --top-k 8 --variant tiny --emit-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --skip-existing
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --summary-only
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --rank-by loss
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-leaderboard outputs/vision/mot_realtime_top3.json
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir outputs/vision/mot_realtime_top3_artifacts
python scripts/mot_zoo.py --recommend realtime --top-k 3 --variant tiny --run-train-cmds --save-artifacts-dir auto
```

## 运行方式

快速冒烟（CPU，几十秒内）：

```bash
python -m tracks.vision.lesson_14_video_mot_basics.train \
  --device cpu --epochs 2 \
  --max-train-batches 5 --max-eval-batches 3 \
  --arch mot2d:sort_tiny \
  --run-name smoke
```

切换不同 MOT 架构族：

```bash
python -m tracks.vision.lesson_14_video_mot_basics.train \
  --device cpu --epochs 2 \
  --max-train-batches 5 --max-eval-batches 3 \
  --arch mot2d:bytetrack_tiny \
  --run-name bytetrack_smoke
```

## 输出产物（统一规范）

`outputs/vision/lesson_14_video_mot_basics/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
