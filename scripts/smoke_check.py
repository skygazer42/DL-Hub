import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    # Also expose the scripts/ directory so the smoke_checks package is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    _ensure_repo_root_on_path()

    from smoke_checks import numpy_core

    # 1)-3) Numpy-only sanity checks.
    results = numpy_core.run()

    # 4) PyTorch tracks sanity checks.
    ran_pytorch = False
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print("smoke_check: torch not available; skipping PyTorch lessons.")
        print(f"- reason: {exc}")
    else:
        ran_pytorch = True

        from smoke_checks import foundations, generative, gnn, llm, nlp, pointcloud, vision

        # Calls preserve the section order of the original monolithic script.
        foundations.check_linear_regression()  # 4.1
        vision.check_mnist_lenet()  # 4.2
        vision.check_synthetic_detection()  # 4.2b
        vision.check_vit_toy_classification()  # 4.2c
        vision.check_swin_toy_classification()  # 4.2d
        pointcloud.check_zoo_toy_classification()  # 4.3
        vision.check_toy_keypoint_regression()  # 4.2e
        vision.check_synthetic_segmentation()  # 4.2f
        vision.check_cnn_backbones()  # 4.2g
        llm.check_toy_causal_lm()  # 4.3
        gnn.run()  # 4.3-4.10
        nlp.run()  # 4.10-4.13
        generative.run()  # 4.13-4.14
        pointcloud.check_pointnet_toy_classification()  # 4.13
        pointcloud.check_dgcnn_toy_classification()  # 4.14
        pointcloud.check_pointnet2_toy_classification()  # 4.15

    print("smoke_check: OK")
    print(f"- LogisticRegression train acc: {results['acc']:.3f}")
    print(f"- WarmupCosine last lr: {results['last_lr']:.6f}")
    print(f"- mean_squared_error(b, 0): {results['mse']:.6f}")
    if ran_pytorch:
        print("- tracks: PyTorch lessons executed")
    else:
        print("- tracks: PyTorch lessons skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
