import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, List
import random

from dc1.image_dataset import ImageDataset

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device  # Get model's device
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # Use full backward hook
    
    def generate_cam(self, input_tensor: torch.Tensor, target_category: int = None) -> np.ndarray:
        """Generate Grad-CAM visualization."""
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        model_output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(model_output)
            
        # Get sigmoid output for binary classification
        prob = torch.sigmoid(model_output)
        
        # Backward pass
        self.model.zero_grad()
        output = model_output[0]
        output.backward()
        
        # Generate CAM
        gradients = self.gradients.squeeze().to(self.device)
        activations = self.activations.squeeze().to(self.device)
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)  # Add small epsilon to avoid division by zero
        
        return cam.cpu().numpy()

def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """Load the trained ResNet-18 model."""
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Flatten(0, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def visualize_gradcam(image: np.ndarray, cam: np.ndarray, output_path: str, class_name: str):
    """Create and save Grad-CAM visualization."""
    # Resize CAM to match image size
    cam = cv2.resize(cam, (128, 128))
    
    # Ensure image is in correct range [0, 255] and type uint8
    image = (image * 255).astype(np.uint8)
    
    # Convert grayscale to RGB for heatmap
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Ensure both images are uint8 type
    image_rgb = image_rgb.astype(np.uint8)
    heatmap = heatmap.astype(np.uint8)
    
    # Combine original image with heatmap
    superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
    
    # Convert BGR to RGB for matplotlib
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(class_name)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cam, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Overlaid Result")
    plt.imshow(superimposed)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_combined_visualization(images: List[np.ndarray], 
                               cams: List[np.ndarray], 
                               class_name: str,
                               output_path: str,
                               probs: List[float]):
    """Create a combined visualization of multiple samples in a grid."""
    num_samples = len(images)
    plt.figure(figsize=(15, 4 * num_samples))
    
    for idx in range(num_samples):
        # Process images like in visualize_gradcam
        image = (images[idx] * 255).astype(np.uint8)
        cam = cv2.resize(cams[idx], (128, 128))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        # Create row for this sample
        plt.subplot(num_samples, 3, idx * 3 + 1)
        plt.title(f"{class_name} (Prob: {probs[idx]:.3f})")
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, idx * 3 + 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(cam, cmap='jet')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, idx * 3 + 3)
        plt.title("Overlaid Result")
        plt.imshow(superimposed)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "model_weights/resnet18_model_03_13_02_09.pt"  # Update with your model path
    output_dir = Path("artifacts/gradcam")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model = load_model(model_path, device)
    model.to(device)
    
    # Initialize Grad-CAM with the last convolutional layer
    grad_cam = GradCAM(model, model.layer4[-1].conv2)
    
    # Load test dataset
    test_dataset = ImageDataset(
        Path("dc1/data/X_test.npy"),
        Path("dc1/data/Y_test_binary.npy")
    )
    
    # Get indices for each class
    normal_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 0]
    pneumo_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 1]
    
    # Number of samples to visualize per class
    num_samples = 5
    
    # Store samples for combined visualization
    normal_samples = {'images': [], 'cams': [], 'probs': []}
    pneumo_samples = {'images': [], 'cams': [], 'probs': []}
    
    # Randomly sample from each class
    for class_label, indices in [(0, normal_indices), (1, pneumo_indices)]:
        selected_indices = random.sample(indices, min(num_samples, len(indices)))
        
        for sample_idx, idx in enumerate(selected_indices):
            image, label = test_dataset[idx]
            
            # Prepare input
            input_tensor = image.unsqueeze(0).to(device)
            
            # Generate Grad-CAM
            cam = grad_cam.generate_cam(input_tensor)
            
            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
            
            # Store samples for combined visualization
            samples_dict = normal_samples if class_label == 0 else pneumo_samples
            samples_dict['images'].append(image.squeeze().numpy())
            samples_dict['cams'].append(cam)
            samples_dict['probs'].append(prob)
            
            # # Create individual visualization - COMMENTED OUT
            # class_name = "Pneumothorax" if class_label == 1 else "No Pneumothorax"
            # output_path = output_dir / f"gradcam_{class_name}_{sample_idx}_pred{prob:.3f}.png"
            # visualize_gradcam(
            #     image.squeeze().numpy(),
            #     cam,
            #     str(output_path),
            #     class_name
            # )
            # 
            # print(f"Generated Grad-CAM for {class_name} sample {sample_idx + 1}")
    
    # Create combined visualizations
    create_combined_visualization(
        normal_samples['images'],
        normal_samples['cams'],
        "No Pneumothorax",
        str(output_dir / "combined_normal.png"),
        normal_samples['probs']
    )
    
    create_combined_visualization(
        pneumo_samples['images'],
        pneumo_samples['cams'],
        "Pneumothorax",
        str(output_dir / "combined_pneumothorax.png"),
        pneumo_samples['probs']
    )
    
    print("Generated combined visualizations")

if __name__ == "__main__":
    main()
