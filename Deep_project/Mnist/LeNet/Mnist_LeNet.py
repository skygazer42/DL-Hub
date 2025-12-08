"""Simple LeNet implementation for MNIST classification.

The script provides a small, configurable training loop that can run on CPU
or GPU. A few convenience arguments make it easy to sanity‑check the pipeline
without running a full 10‑epoch training session.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


@dataclass
class TrainingConfig:
    """Configuration values controlling training behaviour."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.01
    data_dir: str = ".data"
    checkpoint_path: str = "LeNet.pth"
    plot_path: str = "lenet_accuracy.png"
    seed: int = 42
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch PRNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LeNet(nn.Module):
    """Minimal LeNet‑5 style network for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_data_loaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=config.data_dir, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_dir, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            avg_loss = running_loss / 200
            print(f"[batch {batch_idx + 1:05d}] loss: {avg_loss:.4f}")
            running_loss = 0.0

    return running_loss


def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total else 0.0
    print(f"Accuracy on evaluation set: {accuracy * 100:.2f}%")
    return accuracy


def plot_accuracy(epochs: list[int], accuracies: list[float], output_path: str) -> None:
    plt.figure()
    plt.plot(epochs, accuracies, marker="o")
    plt.title("MNIST on LeNet-5")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved accuracy curve to {output_path}")


def train_model(config: TrainingConfig) -> None:
    set_seed(config.seed)

    train_loader, test_loader = get_data_loaders(config)
    model = LeNet().to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    epoch_list: list[int] = []
    acc_list: list[float] = []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
            max_batches=config.max_train_batches,
        )
        accuracy = evaluate(
            model,
            test_loader,
            config.device,
            max_batches=config.max_eval_batches,
        )
        epoch_list.append(epoch + 1)
        acc_list.append(accuracy)

    plot_accuracy(epoch_list, acc_list, config.plot_path)
    avg_acc = np.mean(acc_list)
    print(f"Mean accuracy over {config.epochs} epochs: {avg_acc * 100:.2f}%")
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Model saved to {config.checkpoint_path}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Optimizer learning rate.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit the number of batches per epoch (useful for quick checks).",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Limit the number of evaluation batches (useful for quick checks).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".data",
        help="Directory where MNIST data will be downloaded.",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="LeNet.pth", help="Where to save the trained weights."
    )
    parser.add_argument(
        "--plot-path", type=str, default="lenet_accuracy.png", help="Where to save the accuracy curve."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        plot_path=args.plot_path,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    train_model(parse_args())
