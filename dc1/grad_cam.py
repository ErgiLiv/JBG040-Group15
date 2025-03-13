import torch
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, target_class=None):
        """
        Generate Grad-CAM for a single image
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Optional target class index
        """
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)  # Add batch dimension
            
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)[0]
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        if model_output.shape[0] > 1:
            output_for_class = model_output[0, target_class]  # Take first image if batch
        else:
            output_for_class = model_output[0, target_class]
            
        output_for_class.backward()
        
        # Rest of the method remains the same
        weights = torch.mean(self.gradients, dim=(2, 3))[0, :]
        
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_image.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]
        
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        # Resize to input size
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = torch.nn.functional.interpolate(
            cam, 
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        cam = cam.squeeze()
        
        return cam.cpu().numpy()
