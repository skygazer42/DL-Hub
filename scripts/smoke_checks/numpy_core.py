"""Numpy-only smoke checks (no torch required)."""

import numpy as np


def run() -> dict:
    from ml_algorithms.python.kmeans import KMeans
    from ml_algorithms.python.linear_models import LogisticRegression
    from optimization.python.losses import mean_squared_error
    from optimization.python.lr_schedulers import WarmupCosine
    from optimization.python.metrics import accuracy_score
    from optimization.python.optimizers import Adam

    rng = np.random.default_rng(0)

    # 1) A tiny linear classification sanity check.
    x = rng.normal(size=(256, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    clf = LogisticRegression(learning_rate=0.1, epochs=500).fit(x, y)
    preds = clf.predict(x)
    acc = accuracy_score(y, preds)
    assert acc > 0.85

    # 2) A tiny clustering sanity check.
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
    assert kmeans.cluster_centers_.shape == (3, 2)
    assert kmeans.labels_.shape == (x.shape[0],)

    # 3) Optimizer + scheduler plumbing sanity check.
    params = {"w": rng.normal(size=(3, 3)), "b": np.zeros(3)}
    grads = {"w": np.ones((3, 3)) * 0.1, "b": np.ones(3) * 0.01}
    opt = Adam(learning_rate=1e-3)
    scheduler = WarmupCosine(base_lr=1e-3, warmup_steps=2, max_steps=10)

    for _ in range(3):
        opt.learning_rate = scheduler.step()
        params = opt.step(params, grads)

    mse = mean_squared_error(np.zeros_like(params["b"]), params["b"])
    assert mse >= 0.0

    return {"acc": acc, "last_lr": opt.learning_rate, "mse": mse}
