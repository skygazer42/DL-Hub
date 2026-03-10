import numpy as np

from ml_algorithms.python.kmedoids import KMedoids


def test_kmedoids_recovers_obvious_clusters_and_predicts_new_points() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=(-5.0, 0.0), scale=0.5, size=(80, 2))
    cluster_b = rng.normal(loc=(5.0, 0.0), scale=0.5, size=(80, 2))
    x = np.vstack([cluster_a, cluster_b])

    model = KMedoids(n_clusters=2, random_state=0, max_iter=80).fit(x)

    assert np.unique(model.labels_).size == 2
    assert model.cluster_centers_.shape == (2, 2)
    assert model.medoid_indices_.shape == (2,)
    assert len(set(model.medoid_indices_.tolist())) == 2
    assert np.allclose(model.cluster_centers_, x[model.medoid_indices_])

    true_centers = np.array([[-5.0, 0.0], [5.0, 0.0]])
    distances = np.linalg.norm(
        model.cluster_centers_[:, None, :] - true_centers[None, :, :], axis=2
    )
    nearest_to_true = distances.min(axis=0)
    assert np.all(nearest_to_true < 1.5)

    preds = model.predict(np.array([[-6.0, 0.2], [6.0, -0.1]]))
    assert preds.shape == (2,)
    assert preds[0] != preds[1]
