import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

REGRESSION_IMPORTS = (
    "dlhub.device",
    "ml_algorithms.python.clustering",
    "ml_algorithms.python.gmm",
    "ml_algorithms.python.kmeans",
    "ml_algorithms.python.knn",
    "ml_algorithms.python.linear_models",
    "ml_algorithms.python.mlp",
    "ml_algorithms.python.naive_bayes",
    "ml_algorithms.python.pca",
    "ml_algorithms.python.perceptron",
    "ml_algorithms.python.random_forest",
    "ml_algorithms.python.svm",
    "tracks.gnn.datasets.cora",
    "tracks.gnn.datasets.karate",
    "tracks.gnn.lesson_04_cora_node_classification_gcn.train",
    "tracks.gnn.lesson_05_label_propagation_cora.train",
    "tracks.gnn.lesson_06_graphsage_cora.train",
    "tracks.gnn.lesson_07_sdne_karate_embedding.train",
    "tracks.gnn.lesson_08_line_karate_embedding.train",
    "tracks.gnn.lesson_09_metapath2vec_toy_hetero_embedding.train",
    "tracks.gnn.lesson_10_pinsage_toy_recommender.train",
    "tracks.gnn.lesson_11_rgcn_toy_node_classification.train",
)


@pytest.mark.parametrize("module_name", REGRESSION_IMPORTS)
def test_regression_modules_import_without_nameerror_or_bad_relative_imports(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, (
        f"{module_name} failed to import.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
