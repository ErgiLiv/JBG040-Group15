
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
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
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
from torchvision.models import ResNet18_Weights
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model

def explain_with_lime(model, device, test_sampler, num_samples=5, num_features=10):
    """
    Generates a LIME explanation for an image.

    Parameters:
        model (torch.nn.Module): Trained ResNet model.
        image_path (str): Path to the image.
        num_samples (int): Number of perturbations for LIME.
        num_features (int): Number of important regions to highlight.
    """

    """Generate LIME visualizations for a few test samples"""
 
    
    model.eval()
    explainer = lime_image.LimeImageExplainer()
    test_samples = [batch for batch in test_sampler][:num_samples]  
    plt.figure(figsize=(15, 3*num_samples))
    sample_idx = 0

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    for data, target in test_sampler:
        if sample_idx >= num_samples:
            break

        for i in range(min(len(data), num_samples - sample_idx)):
            single_image = data[i].cpu().numpy().squeeze()  # Convert tensor to numpy
            single_target = target[i].item()

            # Define prediction function for LIME
            def predict_fn(images):
                
                images = torch.tensor(images).permute(0, 3, 1, 2).float()
                images = images.to(device)
                model.to(device)
                if images.shape[1] == 3:
                     images = images.mean(1, keepdim=True)
                outputs = model(images)
                return torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()

            # Generate LIME explanation
            explanation = explainer.explain_instance(
                single_image,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )

            # Get the explanation mask
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=10,
                hide_rest=True
            )

            # Plot original image
            plt.subplot(num_samples, 3, sample_idx*3 + 1)
            plt.imshow(single_image, cmap='gray')
            plt.title(f'Original (True: {single_target})')
            plt.axis('off')

            # Plot LIME mask
            plt.subplot(num_samples, 3, sample_idx*3 + 2)
            plt.imshow(mark_boundaries(temp, mask))
            plt.title('LIME Explanation')
            plt.axis('off')

            # Plot overlay
            plt.subplot(num_samples, 3, sample_idx*3 + 3)
            plt.imshow(single_image, cmap='gray')
            plt.imshow(mask, cmap='jet', alpha=0.5)
            plt.title('Overlay')
            plt.axis('off')

            sample_idx += 1
            if sample_idx >= num_samples:
                break

    plt.tight_layout()
    now = datetime.now()
    plt.savefig(Path("artifacts") / f"lime_{now.month:02}_{now.day}_{now.hour}_{now.minute}.png")
    plt.close()
    
    
    
def load_model(weights_path: str):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify for grayscale
    model.fc = nn.Linear(model.fc.in_features, 6)  # Modify for 6 classes
    model.load_state_dict(torch.load(weights_path, map_location="cpu")) # Load trained weights
    model.eval()
    return model

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load datasets
    train_dataset = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"))

    # Load Pretrained ResNet-18 and modify for 6 classes
    #model = models.resnet18(pretrained=True)
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
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
    if activeloop:
         print("\nGenerating LIME visualizations...")
         explain_with_lime(model, device, test_sampler)
         print("LIME visualizations saved to artifacts folder.")
    
	


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="Number of training iterations", default=10, type=int)
    parser.add_argument("--batch_size", help="Batch size", default=25, type=int)
    parser.add_argument("--balanced_batches", help="Balance batches for class labels", default=True, type=bool)
    args = parser.parse_args()
    main(args)  
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained model separately for LIME
    model_path = sorted(Path("model_weights/").glob("resnet18_model_*.pt"))[-1]  # Get latest model
    model = load_model(str(model_path))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"))
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=True)
    explain_with_lime(model, device, test_sampler)
