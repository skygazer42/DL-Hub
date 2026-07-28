import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    # Also expose the scripts/ directory so the smoke_checks package is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_root_on_path()

    from scripts.lesson_contracts import discover_curated_smoke_lessons, discover_lesson_contracts

    parser = argparse.ArgumentParser(
        description="Run DL-Hub's curated, offline smoke suite across all learning tracks."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List covered lessons without executing the smoke suite.",
    )
    args = parser.parse_args(argv)

    smoke_lessons = sorted(discover_curated_smoke_lessons())
    all_lessons = discover_lesson_contracts()
    if args.list:
        print(f"Curated smoke coverage: {len(smoke_lessons)}/{len(all_lessons)} lessons")
        for track, lesson in smoke_lessons:
            print(f"- {track}/{lesson}")
        return 0

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

        from smoke_checks import (
            foundations,
            generative,
            gnn,
            llm,
            multimodal,
            nlp,
            pointcloud,
            vision,
        )

        # Calls preserve the section order of the original monolithic script.
        foundations.check_linear_regression()  # 4.1
        vision.check_mnist_lenet()  # 4.2
        vision.check_synthetic_detection()  # 4.2b
        vision.check_vit_compact_classification()  # 4.2c
        vision.check_swin_compact_classification()  # 4.2d
        pointcloud.check_zoo_compact_classification()  # 4.3
        vision.check_compact_keypoint_regression()  # 4.2e
        vision.check_synthetic_segmentation()  # 4.2f
        vision.check_cnn_backbones()  # 4.2g
        llm.check_compact_causal_lm()  # 4.3
        gnn.run()  # 4.3-4.10
        nlp.run()  # 4.10-4.13
        generative.run()  # 4.13-4.14
        multimodal.check_clip_compact_retrieval()
        pointcloud.check_pointnet_compact_classification()  # 4.13
        pointcloud.check_dgcnn_compact_classification()  # 4.14
        pointcloud.check_pointnet2_compact_classification()  # 4.15

    print("smoke_check: OK")
    print(f"- curated coverage: {len(smoke_lessons)}/{len(all_lessons)} lessons")
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
