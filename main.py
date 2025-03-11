# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model

# Torch imports
import torch
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import argparse
import plotext
from datetime import datetime
from pathlib import Path
from typing import List


# Focal Loss with Class Weighting
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load datasets
    train_dataset = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"),augment= True)
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"),augment = False)

    # Create model (assumes binary output Net)
    model = Net()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

    # Loss function - Focal Loss
    loss_function = FocalLoss(alpha=0.4, gamma=3.0)

    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    DEBUG = False

    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        summary(model, (1, 128, 128), device=device)
    elif torch.backends.mps.is_available() and not DEBUG:
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        summary(model, (1, 128, 128), device=device)

    train_sampler = BatchSampler(batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    for e in range(n_epochs):
        if activeloop:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss:.4f}\n")

            losses = test_model(model, test_sampler, loss_function, device)
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss:.4f}\n")

            plotext.clf()
            plotext.scatter(mean_losses_train, label="Train")
            plotext.scatter(mean_losses_test, label="Test")
            plotext.title("Train and Test Loss")
            plotext.show()

    now = datetime.now()
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--balanced_batches", default=True, type=bool)

    args = parser.parse_args()
    main(args)
