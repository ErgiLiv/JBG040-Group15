import numpy as np
from pathlib import Path
import cv2
import random
from imgaug import augmenters as iaa

# Load the original label datasets
Y_train_original = np.load(Path("dc1/data/Y_train.npy"))  # Assuming multi-class labels
Y_test_original = np.load(Path("dc1/data/Y_test.npy"))

# Convert to binary: 1 if class is 5 (pneumothorax), else 0
Y_train_binary = (Y_train_original == 5).astype(np.uint8)
Y_test_binary = (Y_test_original == 5).astype(np.uint8)

# Save the binary labels
np.save(Path("dc1/data/Y_train_binary.npy"), Y_train_binary)
np.save(Path("dc1/data/Y_test_binary.npy"), Y_test_binary)

print("Binary label datasets created and saved successfully.")
X_train = np.load(Path("dc1/data/X_train.npy"))  # Shape: (num_samples, H, W, C)
Y_train_binary = np.load(Path("dc1/data/Y_train_binary.npy"))  # Binary labels

# Identify Class 1 (non-pneumothorax) and Class 5 (pneumothorax) samples
class_1_indices = np.where(Y_train_binary == 0)[0]  # Non-pneumothorax (Class 1)
class_5_indices = np.where(Y_train_binary == 1)[0]  # Pneumothorax (Class 5)

# Determine the number of samples needed for balance
num_pneumothorax = len(class_5_indices)
num_non_pneumothorax = len(class_1_indices)
print(num_non_pneumothorax)
print(num_pneumothorax)
# If non-pneumothorax data is less, augmentation is needed
num_augmentations_needed = num_non_pneumothorax - num_pneumothorax

# If already balanced, no augmentation needed
if num_augmentations_needed <= 0:
    print("Dataset is already balanced. No augmentation needed.")
else:
    print(f"Performing augmentation to create {num_augmentations_needed} more non-pneumothorax samples.")

    # Select Class 1 images for augmentation
    class_1_images = X_train[class_1_indices]
    
    # Define augmentation pipeline
    aug_pipeline = iaa.Sequential([
        iaa.Fliplr(0.5),  # Random horizontal flip
        iaa.Affine(rotate=(-10, 10)),  # Random rotation (-10 to 10 degrees)
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Add noise
        iaa.Multiply((0.8, 1.2))  # Adjust brightness randomly
    ])

    # Generate augmented images
    augmented_images = []
    augmented_labels = []

    for _ in range(num_augmentations_needed):
        img_idx = random.choice(class_1_indices)  # Pick a random class 1 image
        img = X_train[img_idx]  # Get the corresponding image
        img_aug = aug_pipeline.augment_image(img)  # Apply augmentation
        augmented_images.append(img_aug)
        augmented_labels.append(0)  # Label remains 0 (non-pneumothorax)

    # Convert to NumPy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    # Merge with original dataset
    X_train_aug = np.concatenate((X_train, augmented_images), axis=0)
    Y_train_aug = np.concatenate((Y_train_binary, augmented_labels), axis=0)

    # Save the new dataset
    np.save(Path("dc1/data/X_train_aug.npy"), X_train_aug)
    np.save(Path("dc1/data/Y_train_aug.npy"), Y_train_aug)

    print("Augmented dataset created and saved successfully.")