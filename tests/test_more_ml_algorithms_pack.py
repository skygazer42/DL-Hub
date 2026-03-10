from __future__ import annotations

import importlib
import importlib.util

import numpy as np


def _require_module(module_name: str):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Expected module to exist: {module_name}"
    return importlib.import_module(module_name)


def test_lasso_regression_fits_sparse_signal_and_shrinks_l1_norm() -> None:
    mod = _require_module("ml_algorithms.python.lasso")
    LassoRegression = getattr(mod, "LassoRegression")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 10))
    true_w = np.zeros(10, dtype=np.float64)
    true_w[[1, 5, 8]] = np.array([3.0, -2.0, 1.5])
    y = x @ true_w + 0.1 * rng.normal(size=x.shape[0])

    ls_w, *_ = np.linalg.lstsq(x, y, rcond=None)

    model = LassoRegression(alpha=0.15, max_iter=2000, tol=1e-6).fit(x, y)
    preds = model.predict(x)
    mse = float(np.mean((preds - y) ** 2))

    assert mse < 0.2
    assert np.linalg.norm(model.weights_, ord=1) < np.linalg.norm(ls_w, ord=1)


def test_elastic_net_regression_is_stable_and_non_trivial() -> None:
    mod = _require_module("ml_algorithms.python.elastic_net")
    ElasticNetRegression = getattr(mod, "ElasticNetRegression")

    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 8))
    true_w = np.array([0.0, 2.0, 0.0, -1.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float64)
    y = x @ true_w + 0.15 * rng.normal(size=x.shape[0])

    model = ElasticNetRegression(alpha=0.2, l1_ratio=0.6, max_iter=2500, tol=1e-6).fit(x, y)
    preds = model.predict(x)
    mse = float(np.mean((preds - y) ** 2))

    assert mse < 0.3
    assert float(np.linalg.norm(model.weights_)) > 0.1


def test_kernel_ridge_regression_rbf_fits_smooth_function() -> None:
    mod = _require_module("ml_algorithms.python.kernel_ridge")
    KernelRidgeRegression = getattr(mod, "KernelRidgeRegression")

    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, size=(80, 1))
    y = np.sin(3.0 * x[:, 0]) + 0.05 * rng.normal(size=x.shape[0])

    model = KernelRidgeRegression(alpha=1e-3, kernel="rbf", gamma=8.0).fit(x, y)
    preds = model.predict(x)
    mse = float(np.mean((preds - y) ** 2))

    assert mse < 0.02


def test_gaussian_process_regressor_predicts_with_reasonable_uncertainty() -> None:
    mod = _require_module("ml_algorithms.python.gaussian_process")
    GaussianProcessRegressor = getattr(mod, "GaussianProcessRegressor")

    rng = np.random.default_rng(3)
    x = np.linspace(-1.0, 1.0, 25)[:, None]
    y = np.cos(2.0 * x[:, 0]) + 0.02 * rng.normal(size=x.shape[0])

    gpr = GaussianProcessRegressor(length_scale=0.4, sigma_f=1.0, noise=1e-6).fit(x, y)

    mean, std = gpr.predict(x, return_std=True)
    assert mean.shape == (x.shape[0],)
    assert std.shape == (x.shape[0],)
    assert np.all(std >= 0.0)

    mse = float(np.mean((mean - y) ** 2))
    assert mse < 0.02


def test_kernel_pca_linear_tracks_major_variance_direction() -> None:
    mod = _require_module("ml_algorithms.python.kernel_pca")
    KernelPCA = getattr(mod, "KernelPCA")

    rng = np.random.default_rng(4)
    z = rng.normal(scale=10.0, size=400)
    x = np.column_stack(
        [
            z,
            rng.normal(scale=1.0, size=z.shape[0]),
            rng.normal(scale=1.0, size=z.shape[0]),
        ]
    )

    kpca = KernelPCA(n_components=1, kernel="linear").fit(x)
    transformed = kpca.transform(x)
    assert transformed.shape == (x.shape[0], 1)

    corr = float(np.corrcoef(transformed[:, 0], z)[0, 1])
    assert abs(corr) > 0.9


def test_mds_recovers_1d_ordering_from_distances() -> None:
    mod = _require_module("ml_algorithms.python.mds")
    MDS = getattr(mod, "MDS")

    t = np.linspace(-2.0, 2.0, 60)
    x = np.column_stack([t, np.zeros_like(t)])

    mds = MDS(n_components=1).fit(x)
    emb = mds.embedding_
    assert emb.shape == (x.shape[0], 1)

    corr = float(np.corrcoef(emb[:, 0], t)[0, 1])
    assert abs(corr) > 0.95


def test_lle_embedding_is_1d_and_correlates_with_latent_coordinate() -> None:
    mod = _require_module("ml_algorithms.python.lle")
    LocallyLinearEmbedding = getattr(mod, "LocallyLinearEmbedding")

    rng = np.random.default_rng(5)
    t = np.linspace(-1.0, 1.0, 120)
    x = np.column_stack([t, 0.05 * rng.normal(size=t.shape[0])])

    lle = LocallyLinearEmbedding(n_neighbors=12, n_components=1, reg=1e-3).fit(x)
    emb = lle.embedding_
    assert emb.shape == (x.shape[0], 1)
    assert np.all(np.isfinite(emb))

    corr = float(np.corrcoef(emb[:, 0], t)[0, 1])
    assert abs(corr) > 0.9


def test_gaussian_kde_density_is_higher_near_data_mass() -> None:
    mod = _require_module("ml_algorithms.python.kde")
    GaussianKDE = getattr(mod, "GaussianKDE")

    rng = np.random.default_rng(6)
    x = rng.normal(size=(400, 1))
    kde = GaussianKDE(bandwidth=0.4).fit(x)

    p0 = float(kde.pdf(np.array([[0.0]]))[0])
    p5 = float(kde.pdf(np.array([[5.0]]))[0])
    assert p0 > p5

