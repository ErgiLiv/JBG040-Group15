import numpy as np
import os
import random
from skimage.io import imsave

# -------------------------------
# CONFIGURATION
# -------------------------------
X_TRAIN_PATH = "dc1/data/X_train.npy"
Y_TRAIN_PATH = "dc1/data/Y_train.npy"
OUTPUT_DIR = "dc1/data"
MAX_SAMPLES = 2500  # Max images per class

# Create folders
os.makedirs(f"{OUTPUT_DIR}/trainA", exist_ok=True)  # Normal X-rays
os.makedirs(f"{OUTPUT_DIR}/trainB", exist_ok=True)  # Pneumothorax X-rays

# -------------------------------
# Load Data
# -------------------------------
X_train = np.load(X_TRAIN_PATH)  # Shape: (N, 1, 128, 128)
Y_train = np.load(Y_TRAIN_PATH)  # Shape: (N,)

# -------------------------------
# Separate Normal & Pneumothorax
# -------------------------------
normal_indices = np.where(Y_train == 0)[0]  # Get indexes of normal cases
pneumo_indices = np.where(Y_train == 1)[0]  # Get indexes of Pneumothorax cases

# Shuffle & limit the Pneumothorax dataset to match Normal
random.shuffle(pneumo_indices)
pneumo_indices = pneumo_indices[:MAX_SAMPLES]

# -------------------------------
# Save Images to trainA & trainB
# -------------------------------
for i, idx in enumerate(normal_indices[:MAX_SAMPLES]):
    img = X_train[idx, 0, :, :]  # Extract grayscale image
    imsave(f"{OUTPUT_DIR}/trainA/{i}.png", img)

for i, idx in enumerate(pneumo_indices):
    img = X_train[idx, 0, :, :]
    imsave(f"{OUTPUT_DIR}/trainB/{i}.png", img)

print(f"✅ Saved {len(normal_indices[:MAX_SAMPLES])} normal X-rays to trainA/")
print(f"✅ Saved {len(pneumo_indices)} Pneumothorax X-rays to trainB/")
