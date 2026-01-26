# Python Optimization Algorithms

This folder contains lightweight NumPy implementations of optimizers, learning-rate schedulers,
losses, and metrics commonly used in deep learning experiments.

## Optimizers
- `SGD`
- `Momentum`
- `RMSProp`
- `Adagrad`
- `Adam`

## Learning-rate schedulers
- `StepDecay`
- `ExponentialDecay`
- `CosineAnnealing`
- `WarmupCosine`

## Losses
- `mean_squared_error`
- `mean_absolute_error`
- `binary_cross_entropy`
- `categorical_cross_entropy`

## Metrics
- `accuracy_score`
- `precision_recall_f1`
- `r2_score`

## Example
```python
import numpy as np
from optimizers import Adam
from lr_schedulers import WarmupCosine
from losses import mean_squared_error
from metrics import r2_score

params = {"w": np.random.randn(3, 3), "b": np.zeros(3)}
opt = Adam(learning_rate=1e-3)

scheduler = WarmupCosine(base_lr=1e-3, warmup_steps=100, max_steps=1000)

for _ in range(10):
    grads = {"w": np.random.randn(3, 3), "b": np.random.randn(3)}
    opt.learning_rate = scheduler.step()
    params = opt.step(params, grads)

preds = params["w"].sum(axis=1)
print(mean_squared_error(np.ones_like(preds), preds))
print(r2_score(np.ones_like(preds), preds))
```
