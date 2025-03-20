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
from dc1.batch_sampler import BatchSampler

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device 
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  
    
    def generate_cam(self, input_tensor: torch.Tensor, target_category: int = None) -> np.ndarray:
        """Generate Grad-CAM visualization."""
        # forward pass
        input_tensor = input_tensor.to(self.device)
        model_output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(model_output)
            
        
        prob = torch.sigmoid(model_output)
        
        # backward pass
        self.model.zero_grad()
        output = model_output[0]
        output.backward()
        
        # generate CAM
        gradients = self.gradients.squeeze().to(self.device)
        activations = self.activations.squeeze().to(self.device)
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)  # normalizes between [0, 1]
        
        return cam.cpu().numpy()

def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """Load the trained ResNet-18 model."""
    model = models.resnet18(pretrained=True)
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
    cam = cv2.resize(cam, (128, 128))
    image = (image * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    image_rgb = image_rgb.astype(np.uint8)
    heatmap = heatmap.astype(np.uint8)
    # combine original image with heatmap
    superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
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
                               labels: List[str],
                               output_path: str,
                               probs: List[float]):
    """Create a combined visualization of multiple samples in a grid."""
    num_samples = len(images)
    plt.figure(figsize=(15, 4 * num_samples))
    
    for idx in range(num_samples):
        # process images 
        image = (images[idx] * 255).astype(np.uint8)
        cam = cv2.resize(cams[idx], (128, 128))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        # create row for this sample
        plt.subplot(num_samples, 3, idx * 3 + 1)
        plt.title(f"{labels[idx]} (Prob: {probs[idx]:.3f})")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = sorted(Path("model_weights/").glob("resnet18_model_*.pt"))[-1]
    #model_path = "model_weights/resnet18_pretrained_binary_aug.pt"
    output_dir = Path("artifacts/gradcam")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # load model
    model = load_model(model_path, device)
    model.to(device)
    
    # initialize Grad-CAM with the last convolutional layer
    grad_cam = GradCAM(model, model.layer4[-1].conv2)
    
    # load test dataset and create balanced sampler
    test_dataset = ImageDataset(
        Path("dc1/data/X_test.npy"),
        Path("dc1/data/Y_test_binary.npy")
    )
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=True)
    
    num_samples = 3
    samples = {'images': [], 'cams': [], 'probs': [], 'labels': []}
    sample_idx = 0
    
    for data, target in test_sampler:
        if sample_idx >= num_samples:
            break
            
        for i in range(min(len(data), num_samples - sample_idx)):
            image = data[i]
            label = target[i].item()
            
            input_tensor = image.unsqueeze(0).to(device)
            
            # generate Grad-CAM
            cam = grad_cam.generate_cam(input_tensor)
            
            # model prediction
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
            
            samples['images'].append(image.squeeze().numpy())
            samples['cams'].append(cam)
            samples['probs'].append(prob)
            samples['labels'].append("Pneumothorax" if label == 1 else "No Pneumothorax")
            
            sample_idx += 1
            if sample_idx >= num_samples:
                break
    
    create_combined_visualization(
        samples['images'],
        samples['cams'],
        samples['labels'],
        str(output_dir / "gradcam_visualization.png"),
        samples['probs']
    )
    
    print("Generated Grad-CAM visualization")

if __name__ == "__main__":
    main()
