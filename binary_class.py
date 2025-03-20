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
    # Load datasets
    train_dataset = ImageDataset(Path("dc1/data/X_train_aug.npy"), Path("dc1/data/Y_train_aug.npy"))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test_binary.npy"))

    # Load ResNet-18 (NO pretrained weights)
    model = models.resnet18(pretrained=True)

    # Modify first convolution layer for grayscale images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify final fully connected layer for binary classification (1 output neuron)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Flatten(0, 1)  # Flatten output to match target dimensions
    )

    # Optimizer & Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.BCEWithLogitsLoss()  # Binary classification loss

    # GPU or CPU selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Batch sampling
    train_sampler = BatchSampler(batch_size=args.batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train = []
    mean_losses_test = []
    all_predictions = []
    all_labels = []
    
    for e in range(args.nb_epochs):
        if activeloop:
            # Training
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            mean_loss_train = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss_train)  
            print(f"\nEpoch {e + 1} training done, loss: {mean_loss_train}\n")

            # Testing
            losses, predictions, labels = test_model(model, test_sampler, loss_function, device)
            mean_loss_test = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss_test)  
            if e == args.nb_epochs - 1:
                all_predictions = predictions
                all_labels = labels

            print(f"Epoch {e + 1} testing done, loss: {mean_loss_test}\n")

    # Save model weights
    now = datetime.now()
    Path("model_weights/").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"model_weights/resnet18_model_{now.strftime('%m_%d_%H_%M')}.pt")

    # Plot losses
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(1, args.nb_epochs + 1), mean_losses_train, label="Train", color="blue")
    ax2.plot(range(1, args.nb_epochs + 1), mean_losses_test, label="Test", color="red")
    fig.legend()
    Path("artifacts/").mkdir(exist_ok=True)
    fig.savefig(Path("artifacts") / f"resnet18_session_{now.strftime('%m_%d_%H_%M')}.png")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path("artifacts") / f"confusion_matrix_{now.strftime('%m_%d_%H_%M')}.png")
    plt.close()

    # Classification Report
    print(classification_report(all_labels, all_predictions, target_names=["Non-Pneumothorax", "Pneumothorax"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size")
    parser.add_argument("--balanced_batches", action="store_true", help="Balance batches for class labels")
    args = parser.parse_args()

    main(args)


