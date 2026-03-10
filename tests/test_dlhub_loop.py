import pytest

torch = pytest.importorskip("torch")


def test_fit_classifier_improves_on_toy_data() -> None:
    from dlhub.data.toy import ToyClassificationConfig, make_toy_classification_dataloaders
    from dlhub.seed import set_seed
    from dlhub.training.loop import evaluate_classifier, fit_classifier

    set_seed(0)
    train_loader, val_loader = make_toy_classification_dataloaders(
        ToyClassificationConfig(
            num_samples=256,
            num_features=2,
            noise_std=0.0,
            val_fraction=0.2,
            seed=0,
        ),
        batch_size=32,
        num_workers=0,
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)

    before = evaluate_classifier(
        model=model, loader=val_loader, criterion=criterion, device=device, max_batches=10
    )

    for _ in range(15):
        fit_classifier(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

    after = evaluate_classifier(
        model=model, loader=val_loader, criterion=criterion, device=device, max_batches=10
    )

    assert after.accuracy > 0.9
    assert after.loss < before.loss
