import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model



def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    train_dataset = ImageDataset(Path("dc1/data/X_train_aug.npy"), Path("dc1/data/Y_train_aug.npy"))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test_binary.npy"))

    # pretrained ResNet18 model
    model = models.resnet18(pretrained=True)


    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # modify final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Flatten(0, 1)  
    )

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.BCEWithLogitsLoss()  # binary classification loss

    # GPU or CPU selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # batch sampling
    train_sampler = BatchSampler(batch_size=args.batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train = []
    mean_losses_test = []
    all_predictions = []
    all_labels = []
    
    for e in range(args.nb_epochs):
        if activeloop:
            # training
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            mean_loss_train = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss_train)  
            print(f"\nEpoch {e + 1} training done, loss: {mean_loss_train}\n")

            # testing
            losses, predictions, labels = test_model(model, test_sampler, loss_function, device)
            mean_loss_test = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss_test)  
            if e == args.nb_epochs - 1:
                all_predictions = predictions
                all_labels = labels

            print(f"Epoch {e + 1} testing done, loss: {mean_loss_test}\n")

    # save model weights
    now = datetime.now()
    Path("model_weights/").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"model_weights/resnet18_model_{now.strftime('%m_%d_%H_%M')}.pt")

    # plot losses
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(1, args.nb_epochs + 1), mean_losses_train, label="Train", color="blue")
    ax2.plot(range(1, args.nb_epochs + 1), mean_losses_test, label="Test", color="red")
    fig.legend()
    Path("artifacts/").mkdir(exist_ok=True)
    fig.savefig(Path("artifacts") / f"resnet18_session_{now.strftime('%m_%d_%H_%M')}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size")
    parser.add_argument("--balanced_batches", action="store_true", help="Balance batches for class labels")
    args = parser.parse_args()

    main(args)


