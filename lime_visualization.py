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

def explain_with_lime(model, device, test_sampler, num_samples=3, num_features=100):
    # create directories if they don't exist
    artifacts_dir = Path("artifacts")
    lime_dir = artifacts_dir / "lime"
    artifacts_dir.mkdir(exist_ok=True)
    lime_dir.mkdir(exist_ok=True)
    
    model.eval()
    explainer = lime_image.LimeImageExplainer()
    plt.figure(figsize=(15, 4 * num_samples))
    sample_idx = 0

    for data, target in test_sampler:
        if sample_idx >= num_samples:
            break

        for i in range(min(len(data), num_samples - sample_idx)):
            # prepare the image
            single_image = data[i].cpu().numpy().squeeze()
            single_target = target[i].item()
            
            # ensure proper scaling for LIME
            single_image = (single_image - single_image.min()) / (single_image.max() - single_image.min())
            
            # convert to 3 channels for LIME
            single_image_3ch = np.stack([single_image]*3, axis=-1)

            def predict_fn(images):
                batch = torch.tensor(images, dtype=torch.float32)
                batch = batch.permute(0, 3, 1, 2)
                
                # convert to grayscale by taking mean across channels
                batch = batch.mean(dim=1, keepdim=True)
                
                batch = batch.to(device)
                model.to(device)
                
                with torch.no_grad():
                    output = model(batch)
                    probs = torch.sigmoid(output)
                    # stack probabilities for binary classification
                    return np.column_stack([1-probs.cpu().numpy(), probs.cpu().numpy()])

            # generate LIME explanation with adjusted parameters
            explanation = explainer.explain_instance(
                single_image_3ch,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=2000,
                num_features=num_features,
                random_seed=42
            )

            # get the explanation mask for the positive class (index 1)
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=num_features,
                hide_rest=True
            )

            # model prediction and probability
            with torch.no_grad():
                pred = model(data[i].unsqueeze(0).to(device))
                prob = torch.sigmoid(pred).item()
            
            class_name = "Pneumothorax" if single_target == 1 else "No Pneumothorax"
            
            # plot original image with class name and probability
            plt.subplot(num_samples, 3, sample_idx*3 + 1)
            plt.imshow(single_image, cmap='gray')
            plt.title(f'{class_name} (Prob: {prob:.3f})')
            plt.axis('off')

            # plot LIME mask
            plt.subplot(num_samples, 3, sample_idx*3 + 2)
            plt.imshow(mark_boundaries(temp, mask))
            plt.title('LIME Explanation')
            plt.axis('off')

            # plot overlay
            plt.subplot(num_samples, 3, sample_idx*3 + 3)
            plt.imshow(single_image, cmap='gray')
            plt.imshow(mark_boundaries(temp, mask))
            plt.imshow(mask, cmap='jet', alpha=0.4)
            plt.title('Overlaid result')
            plt.axis('off')

            sample_idx += 1
            if sample_idx >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(lime_dir / f"lime_visualization.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    
    
def load_model(weights_path: str):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1)
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load trained model
    model_path = sorted(Path("model_weights/").glob("resnet18_model_*.pt"))[-1]
    #model_path = "model_weights/resnet18_pretrained_binary_aug.pt" # ATTENTION! Please change this model path accordingly.
    model = load_model(str(model_path))
    
    # create test dataset and sampler
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test_binary.npy"))
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=True)
    
    print("\nGenerating LIME visualizations...")
    explain_with_lime(model, device, test_sampler)
    print("LIME visualizations saved to artifacts/lime folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="Number of training iterations", default=10, type=int)
    parser.add_argument("--batch_size", help="Batch size", default=25, type=int)
    parser.add_argument("--balanced_batches", help="Balance batches for class labels", default=True, type=bool)
    args = parser.parse_args()
    main(args)
