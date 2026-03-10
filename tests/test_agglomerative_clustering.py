import numpy as np
import pytest

from ml_algorithms.python.clustering import AgglomerativeClustering


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_agglomerative_clustering_recovers_obvious_clusters(linkage: str) -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=(-6.0, 0.0), scale=0.4, size=(80, 2))
    cluster_b = rng.normal(loc=(6.0, 0.0), scale=0.4, size=(80, 2))
    x = np.vstack([cluster_a, cluster_b])

    model = AgglomerativeClustering(n_clusters=2, linkage=linkage).fit(x)

    labels = model.labels_
    assert np.unique(labels).size == 2

    centers = np.vstack([x[labels == idx].mean(axis=0) for idx in np.unique(labels)])
    true_centers = np.array([[-6.0, 0.0], [6.0, 0.0]])
    distances = np.linalg.norm(centers[:, None, :] - true_centers[None, :, :], axis=2)
    nearest_to_true = distances.min(axis=0)
    assert np.all(nearest_to_true < 1.0)


def test_agglomerative_clustering_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(60, 3))

    model1 = AgglomerativeClustering(n_clusters=3, linkage="average").fit(x)
    model2 = AgglomerativeClustering(n_clusters=3, linkage="average").fit(x)
    assert np.array_equal(model1.labels_, model2.labels_)


def test_agglomerative_clustering_validates_inputs() -> None:
    x = np.zeros((5, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=1, linkage="single").fit(x)
    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=10, linkage="single").fit(x)
    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=2, linkage="unknown").fit(x)
