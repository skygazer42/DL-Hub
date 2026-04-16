# Lesson 07: Toy Rectified Flow

This lesson continues from toy flow matching with a rectified-flow framing:

- draw endpoint pairs `(x_noise, x_data)` where `x_data` comes from a synthetic image dataset
- train a tiny time-conditioned conv net on interpolation states
- predict the straight-line velocity field needed to transport noise into data

The setup is intentionally small and CPU-friendly. It keeps the continuous-time view introduced in
flow matching while presenting rectified flow as a practical "straight trajectories first" baseline.

## Run

```bash
python -m tracks.generative.lesson_07_toy_rectified_flow.train --epochs 1 --num-samples 48 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs

`outputs/generative/lesson_07_toy_rectified_flow/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `interp.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

If `torchvision` is available, `samples.png` is also written.
