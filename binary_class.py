import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from dc1.resnet18 import CustomResNet
from dc1.Focal_loss import FocalLoss
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model

def relabel_dataset(input_file: str, output_file: str, pneumothorax_class: int = 5):
    """Convert multi-class labels to binary classification (1: Pneumothorax, 0: Other)."""
    labels = np.load(input_file)
    new_labels = np.where(labels == pneumothorax_class, 1, 0)
    np.save(output_file, new_labels)
    print(f"Relabeled dataset saved to {output_file}")

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    



    train_dataset = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train_binary.npy"))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test_binary.npy"))

    

    # Load ResNet-18 (NO pretrained weights)
    model = CustomResNet(num_classes=1)

    # Modify first convolution layer for grayscale images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify final fully connected layer for binary classification (1 output neuron)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

    model.to(device)
    from collections import Counter

    # Count occurrences of each class in training labels
    class_counts = Counter(train_dataset.targets)
    #total_samples = sum(class_counts.values())

    # Compute pos_weight for BCEWithLogitsLoss
    #pos_weight = torch.tensor([class_counts[0] / (class_counts[1] + 1e-6)]).to(device)
    pneumothorax_weight = 2.0  
    pos_weight = torch.tensor([pneumothorax_weight]).to(device)


    # Apply weighted loss function
    #loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]).to(device))
    loss_function = FocalLoss(alpha=0.75, gamma=2).to(device)
    
    weights = [1.5 / class_counts[label] if label == 1 else 1.0 / class_counts[label] for label in train_dataset.targets]


    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)



    # Batch sampling
    #train_sampler = BatchSampler(batch_size=args.batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    #test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train = []
    mean_losses_test = []
    all_predictions = []
    all_labels = []
    
    for e in range(args.nb_epochs):
        if activeloop:
            # Training
            losses = train_model(model, train_loader, optimizer, loss_function, device)
            mean_loss_train = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss_train)  
            print(f"\nEpoch {e + 1} training done, loss: {mean_loss_train}\n")

            # Testing
            losses, predictions, labels, _ = test_model(model, test_loader, loss_function, device)
            mean_loss_test = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss_test)  
            if e == args.nb_epochs - 1:
                #all_predictions = (torch.sigmoid(torch.tensor(all_predictions)) > 0.5).int().numpy()
                all_predictions=predictions

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

    # Convert logits to probabilities using sigmoid
    probabilities = torch.sigmoid(torch.tensor(all_predictions))

    threshold = 0.7
    all_predictions = (probabilities > threshold).int().numpy()


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


