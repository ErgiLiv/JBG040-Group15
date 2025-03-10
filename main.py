import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import argparse
import plotext
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load datasets
    train_dataset = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"))

    # Load Pretrained ResNet-18 and modify for 6 classes
    model = models.resnet18(pretrained=True)
    
    # Modify first convolution layer to accept grayscale images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)  # 6 classes

    # Initialize SGD optimizer with momentum, weight decay and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Learning Rate scheduler to reduce the learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    loss_function = nn.CrossEntropyLoss()

    # Training setup
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # GPU or CPU selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    summary(model, (1, 128, 128), device=device)

    # Batch sampling
    train_sampler = BatchSampler(batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train = []
    mean_losses_test = []
    all_predictions = []
    all_labels = []
    
    for e in range(n_epochs):
        if activeloop:
            # Training
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss: {mean_loss}\n")

            # Testing
            losses, predictions, labels, _ = test_model(model, test_sampler, loss_function, device)
            if e == n_epochs - 1:
                all_predictions = predictions
                all_labels = labels

            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss: {mean_loss}\n")

            # Update learning rate scheduler
            scheduler.step(mean_loss)  # Step on test loss

            # Live loss plot
            plotext.clf()
            plotext.scatter(mean_losses_train, label="Train Loss")
            plotext.scatter(mean_losses_test, label="Test Loss")
            plotext.title("Train vs Test Loss")
            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])
            plotext.show()

    # Save model weights
    now = datetime.now()
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    torch.save(model.state_dict(), f"model_weights/resnet18_model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.pt")

    # Plot losses
    plt.figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(1, 1 + n_epochs), [x for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))
    fig.savefig(Path("artifacts") / f"resnet18_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path("artifacts") / f"confusion_matrix_{now.month:02}_{now.day}_{now.hour}_{now.minute}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="Number of training iterations", default=10, type=int)
    parser.add_argument("--batch_size", help="Batch size", default=25, type=int)
    parser.add_argument("--balanced_batches", help="Balance batches for class labels", default=True, type=bool)
    args = parser.parse_args()

    main(args)