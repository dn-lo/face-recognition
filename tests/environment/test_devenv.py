import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FullyConnectedNet(pl.LightningModule):
    def __init__(self, image_size: int, number_classes: int) -> None:
        super().__init__()
        linear_channels = 128
        image_flatten_size = image_size * image_size

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            in_features=image_flatten_size, out_features=linear_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            in_features=linear_channels, out_features=number_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x

    def step(self, batch: tuple, batch_idx: int, stage: str = "train"):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple, batch_idx: int):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def test_numpy():
    """Test to ensure numpy is working correctly."""
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_pytorch_lightning():
    """Test MNIST classification with PyTorch Lightning."""
    # MNIST dataset setup
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Get image size and number of classes
    image_size = train_dataset[0][0].shape[1]
    number_classes = len(train_dataset.classes)

    # Initialize model and trainer
    model = FullyConnectedNet(image_size=image_size,
                              number_classes=number_classes)
    trainer = pl.Trainer(
        max_epochs=2, enable_checkpointing=False, logger=False, accelerator="auto")

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Verify model learned something
    print(trainer.callback_metrics)
    assert trainer.callback_metrics["val_acc"] > 0.85
