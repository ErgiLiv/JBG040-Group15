import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, augment: bool = False):
        self.images = np.load(x_path)
        self.labels = np.load(y_path)
        self.targets = self.labels
        self.augment = augment

        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Check initial shape (should be [1, 128, 128])
        assert image.shape == (1, 128, 128), f"Unexpected shape at index {idx}: {image.shape}"

        # Convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # Apply augmentations if in training mode
        if self.augment:
            image_tensor = image_tensor.squeeze(0)  # Make (128, 128)
            image_tensor = T.ToPILImage()(image_tensor)
            image_tensor = self.augmentations(image_tensor)
            image_tensor = T.ToTensor()(image_tensor)  # Back to (1, 128, 128) automatically

        return image_tensor, label
