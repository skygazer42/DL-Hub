import numpy as np

from ml_algorithms.python.kmeans import KMeans
from ml_algorithms.python.pca import PCA


def test_kmeans_recovers_obvious_clusters() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=(-5.0, 0.0), scale=0.5, size=(120, 2))
    cluster_b = rng.normal(loc=(5.0, 0.0), scale=0.5, size=(120, 2))
    x = np.vstack([cluster_a, cluster_b])

    model = KMeans(n_clusters=2, random_state=0).fit(x)

    assert np.unique(model.labels_).size == 2

    true_centers = np.array([[-5.0, 0.0], [5.0, 0.0]])
    distances = np.linalg.norm(
        model.cluster_centers_[:, None, :] - true_centers[None, :, :], axis=2
    )
    nearest_to_true = distances.min(axis=0)
    assert np.all(nearest_to_true < 1.0)


def test_pca_reduces_dimension_and_captures_major_variance_axis() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(scale=10.0, size=600)
    x = np.column_stack(
        [
            z,
            rng.normal(scale=1.0, size=z.shape[0]),
            rng.normal(scale=1.0, size=z.shape[0]),
        ]
    )

    pca = PCA(n_components=1).fit(x)
    transformed = pca.transform(x)

    assert transformed.shape == (x.shape[0], 1)
    assert pca.explained_variance_.shape == (1,)
    assert pca.explained_variance_[0] > 50.0

    correlation = np.corrcoef(transformed[:, 0], z)[0, 1]
    assert abs(correlation) > 0.9
