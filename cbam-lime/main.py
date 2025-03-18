from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model
from lime import lime_image
from skimage.segmentation import mark_boundaries

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from skimage.segmentation import slic, quickshift

def custom_segmentation(image):
    return slic(image, n_segments=22, compactness=10, sigma=1)
# Focal Loss with Class Weighting
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def explain_with_lime(model, device, test_sampler, num_samples=5, num_features=15):
    """Generate LIME explanations for a few test samples"""
    model.eval()
    explainer = lime_image.LimeImageExplainer()
    test_samples = [batch for batch in test_sampler][:num_samples]  

    plt.figure(figsize=(15, 3 * num_samples))
    sample_idx = 0

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Added normalization to avoid extreme contrast
    ])

    for data, target in test_samples:
        if sample_idx >= num_samples:
            break

        for i in range(min(len(data), num_samples - sample_idx)):
            single_image = data[i].cpu().numpy()
            if single_image.shape[0] == 1:  # Ensure proper shape handling
                single_image = single_image.squeeze(0)
            
            single_target = target[i].item()
            
            def predict_fn(images):
                images = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)
                model.to(device)
                if images.shape[1] == 3:
                    images = images.mean(1, keepdim=True)  # Convert back to grayscale for model
                outputs = model(images)
                return torch.sigmoid(outputs).detach().cpu().numpy()

            explanation = explainer.explain_instance(
                single_image,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000,
                segmentation_fn=custom_segmentation
            )

            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label, positive_only=True, num_features=num_features, hide_rest=True
            )

            plt.subplot(num_samples, 3, sample_idx * 3 + 1)
            plt.imshow(single_image, cmap='gray')
            plt.title(f'Original (True: {single_target})')
            plt.axis('off')

            plt.subplot(num_samples, 3, sample_idx * 3 + 2)
            plt.imshow(mark_boundaries(temp, mask))
            plt.title('LIME Explanation')
            plt.axis('off')

            plt.subplot(num_samples, 3, sample_idx * 3 + 3)
            plt.imshow(single_image, cmap='gray')
            plt.imshow(mask, cmap='jet', alpha=0.4)
            plt.title('Overlay')
            plt.axis('off')

            sample_idx += 1
            if sample_idx >= num_samples:
                break

    plt.tight_layout()
    now = datetime.now()
    plt.savefig(Path("artifacts") / f"lime_{now.month:02}_{now.day}_{now.hour}_{now.minute}.png")
    plt.close()


def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train_binary.npy"), augment=True)
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test_binary.npy"), augment=False)

    model = Net()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = FocalLoss(alpha=0.25, gamma=2.0)

    train_sampler = BatchSampler(batch_size=args.batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    if args.nb_epochs > 0:
        train_model(model, train_sampler, optimizer, loss_function, device)

    # LIME visualization
    print("\nGenerating LIME visualizations...")
    explain_with_lime(model, device, test_sampler)
    print("LIME visualizations saved to artifacts folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--balanced_batches", default=True, type=bool)

    args = parser.parse_args()
    main(args)
