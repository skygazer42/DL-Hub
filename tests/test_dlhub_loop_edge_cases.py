import pytest

from dlhub.training.hooks import BatchLog, Hook
from dlhub.training.loop import (
    RegressionStats,
    SegmentationStats,
    TokenStats,
    TrainStats,
    evaluate_binary_segmentation,
    evaluate_classifier,
    evaluate_regression,
    evaluate_token_classifier,
    fit_binary_segmentation,
    fit_classifier,
    fit_regression,
    fit_token_classifier,
)

torch = pytest.importorskip("torch")


def _classifier_components():
    model = torch.nn.Linear(2, 2)
    loader = [(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))]
    criterion = torch.nn.CrossEntropyLoss()
    return model, loader, criterion


@pytest.mark.parametrize(
    ("max_batches", "error_type"),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_max_batches_rejects_invalid_values(
    max_batches: object, error_type: type[Exception]
) -> None:
    model, loader, criterion = _classifier_components()

    with pytest.raises(error_type, match="max_batches"):
        evaluate_classifier(
            model=model,
            loader=loader,
            criterion=criterion,
            device=torch.device("cpu"),
            max_batches=max_batches,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("loader,max_batches", [([], None), (_classifier_components()[1], 0)])
def test_classifier_empty_selection_returns_zero_stats(loader, max_batches) -> None:
    model, _, criterion = _classifier_components()

    stats = evaluate_classifier(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        max_batches=max_batches,
    )

    assert stats == TrainStats(loss=0.0, accuracy=0.0)


class _NeverIteratedLoader:
    def __iter__(self):
        raise AssertionError("max_batches=0 must not iterate the loader")


class _NeverCalledModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        del inputs
        raise AssertionError("max_batches=0 must not run model.forward")


class _NeverCalledCriterion(torch.nn.Module):
    def forward(self, outputs, targets):
        del outputs, targets
        raise AssertionError("max_batches=0 must not call the criterion")


class _NeverUsedOptimizer:
    def zero_grad(self, *, set_to_none: bool) -> None:
        del set_to_none
        raise AssertionError("max_batches=0 must not touch the optimizer")

    def step(self) -> None:
        raise AssertionError("max_batches=0 must not touch the optimizer")


@pytest.mark.parametrize(
    ("loop", "expected", "training"),
    [
        (fit_classifier, TrainStats(loss=0.0, accuracy=0.0), True),
        (evaluate_classifier, TrainStats(loss=0.0, accuracy=0.0), False),
        (fit_token_classifier, TokenStats(loss=0.0, accuracy=0.0), True),
        (evaluate_token_classifier, TokenStats(loss=0.0, accuracy=0.0), False),
        (fit_binary_segmentation, SegmentationStats(loss=0.0, iou=0.0), True),
        (evaluate_binary_segmentation, SegmentationStats(loss=0.0, iou=0.0), False),
        (fit_regression, RegressionStats(loss=0.0), True),
        (evaluate_regression, RegressionStats(loss=0.0), False),
    ],
)
def test_zero_max_batches_does_not_touch_data_or_compute(loop, expected, training: bool) -> None:
    model = _NeverCalledModel()
    model.train(not training)
    kwargs = {
        "model": model,
        "loader": _NeverIteratedLoader(),
        "criterion": _NeverCalledCriterion(),
        "device": torch.device("cpu"),
        "max_batches": 0,
        "hooks": [_ExplodingHook()],
    }
    if loop.__name__.startswith("fit_"):
        kwargs["optimizer"] = _NeverUsedOptimizer()

    stats = loop(**kwargs)

    assert stats == expected
    assert model.training is training


class _CollectingHook(Hook):
    def __init__(self) -> None:
        self.logs: list[BatchLog] = []

    def on_batch_end(self, log: BatchLog) -> None:
        self.logs.append(log)


def test_all_ignored_token_batch_is_a_true_noop() -> None:
    model = torch.nn.Linear(3, 4)
    inputs = torch.randn(2, 5, 3)
    targets = torch.full((2, 5), -100, dtype=torch.long)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1.0)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    hook = _CollectingHook()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    stats = fit_token_classifier(
        model=model,
        loader=[(inputs, targets)],
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        ignore_index=-100,
        hooks=[hook],
    )

    assert stats.loss == 0.0
    assert stats.accuracy == 0.0
    assert hook.logs == [BatchLog(stage="train", batch_idx=0, loss=0.0, accuracy=0.0)]
    for expected, actual in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None


def test_binary_segmentation_counts_two_empty_masks_as_exact_match() -> None:
    model = torch.nn.Identity()
    logits = torch.full((2, 1, 3, 3), -20.0)
    targets = torch.zeros_like(logits)

    stats = evaluate_binary_segmentation(
        model=model,
        loader=[(logits, targets)],
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
    )

    assert stats.iou == pytest.approx(1.0)


class _ExplodingHook(Hook):
    def on_batch_end(self, log: BatchLog) -> None:
        del log
        raise ValueError("hook exploded")


def test_hook_failure_includes_hook_stage_and_batch_context() -> None:
    model, loader, criterion = _classifier_components()

    with pytest.raises(
        RuntimeError,
        match=r"_ExplodingHook.*stage='eval'.*batch_idx=0",
    ) as caught:
        evaluate_classifier(
            model=model,
            loader=loader,
            criterion=criterion,
            device=torch.device("cpu"),
            hooks=[_ExplodingHook()],
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "hook exploded"
