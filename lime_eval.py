
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
            single_image = data[i].cpu().numpy().squeeze()  
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
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
    model.fc = nn.Linear(model.fc.in_features, 6)  
    model.load_state_dict(torch.load(weights_path, map_location="cpu")) 
    return model

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
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
    model_path = sorted(Path("model_weights/").glob("resnet18_model_*.pt"))[-1]  
    model = load_model(str(model_path))
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"))
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=True)
    explain_with_lime(model, device, test_sampler)
