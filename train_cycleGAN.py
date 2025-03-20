import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import subprocess
import torch  # Add torch import for CUDA detection

# -------------------------------
# CONFIGURATION
# -------------------------------
DATASET_DIR = "dc1/data"  # ✅ Ensure correct path
TRAIN_A_DIR = f"{DATASET_DIR}/TrainA"  # ✅ Update path for TrainA
TRAIN_B_DIR = f"{DATASET_DIR}/TrainB"  # ✅ Update path for TrainB

MODEL_NAME = "cyclegan_chestxray"
OUTPUT_DIR = "dc1/data"
SYNTHETIC_DIR = f"{OUTPUT_DIR}/{MODEL_NAME}/test_latest/images"
TRAINING_EPOCHS = 10  # Reduce for CPU training

# Paths for dataset
X_TRAIN_PATH = "dc1/data/X_train.npy"
Y_TRAIN_PATH = "dc1/data/Y_train.npy"
OUTPUT_X_PATH = "X_train_aug.npy"
OUTPUT_Y_PATH = "Y_train_aug.npy"

# Detect CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_ids = "0" if device == "cuda" else "-1"
print(f"🔧 Using device: {device}")

# -------------------------------
# Function to Run Shell Commands
# -------------------------------
def run_command(command):
    print(f"🔄 Running: {' '.join(command)}")
    subprocess.run(command)

# -------------------------------
# Step 1: Clone CycleGAN (If Not Already Done)
# -------------------------------
if not os.path.exists("pytorch-CycleGAN-and-pix2pix"):
    print("🔄 Cloning CycleGAN repository...")
    run_command(["git", "clone", "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git"])

# -------------------------------
# Step 2: Train CycleGAN on CPU (Using Correct Paths)
# -------------------------------
os.chdir("pytorch-CycleGAN-and-pix2pix")  # Move into CycleGAN repo

# Check if model exists and is properly trained
model_exists = os.path.exists(f"checkpoints/{MODEL_NAME}")
checkpoints_exist = os.path.exists(f"checkpoints/{MODEL_NAME}/latest_net_G.pth")

if not model_exists or not checkpoints_exist:
    print("🔄 Training CycleGAN (this will take time)...")
    train_command = [
        "python", "train.py",
        "--dataroot", f"../{DATASET_DIR}",
        "--name", MODEL_NAME,
        "--model", "cycle_gan",
        "--gpu_ids", gpu_ids,  # Use detected GPU if available
        "--input_nc", "1",
        "--output_nc", "1",
        "--n_epochs", str(TRAINING_EPOCHS),
        "--n_epochs_decay", str(TRAINING_EPOCHS),
        "--direction", "AtoB",
        "--dataset_mode", "unaligned",
        "--load_size", "128",
        "--crop_size", "128",
        "--display_id", "-1",  # Disable visdom
        "--no_html",  # Disable HTML visualization
        "--verbose",  # Add verbose output
        "--max_dataset_size", "300",  # Limit dataset size to control iterations
        "--batch_size", "1"  # Ensure each iteration processes one image
    ]
    
    try:
        run_command(train_command)
        
        # Verify training completed successfully
        if not os.path.exists(f"checkpoints/{MODEL_NAME}/latest_net_G.pth"):
            raise FileNotFoundError(f"❌ Error: Training failed. No checkpoints were created for {MODEL_NAME}.")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise
else:
    print("✅ CycleGAN already trained. Skipping training.")

# -------------------------------
# Step 3: Generate Synthetic Pneumothorax X-rays on CPU (Using Correct Paths)
# -------------------------------
print("🔄 Generating synthetic Pneumothorax images...")

run_command([
    "python", "test.py",
    "--dataroot", f"../{DATASET_DIR}",  # ✅ Ensure correct dataset path
    "--name", MODEL_NAME,
    "--model", "test",
    "--gpu_ids", gpu_ids,  # Use detected GPU if available
    "--input_nc", "1",  # ✅ Fix for grayscale images
    "--output_nc", "1",  # ✅ Fix for grayscale images
    "--direction", "AtoB",  # ✅ Ensure correct transformation direction
])

os.chdir("..")  # Back to main folder

# -------------------------------
# Step 4: Convert Generated Images into .npy Format
# -------------------------------
print("🔄 Converting synthetic images to NumPy format...")

# Ensure the directory exists
if not os.path.exists(SYNTHETIC_DIR):
    raise FileNotFoundError(f"❌ Error: No generated images found in {SYNTHETIC_DIR}. Training may have failed.")

synthetic_images = []
for file_name in os.listdir(SYNTHETIC_DIR):
    if "fake" in file_name and file_name.endswith(".png"):
        img_path = os.path.join(SYNTHETIC_DIR, file_name)
        img = imread(img_path, as_gray=True)
        img_resized = resize(img, (128, 128), anti_aliasing=True)

        # Normalize pixel values
        img_resized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))

        synthetic_images.append(img_resized)

synthetic_images = np.array(synthetic_images).reshape(-1, 1, 128, 128)
synthetic_labels = np.ones(len(synthetic_images))  # Label all as Pneumothorax

np.save("X_synthetic.npy", synthetic_images)
np.save("Y_synthetic.npy", synthetic_labels)
print(f"✅ Saved synthetic dataset: X_synthetic.npy ({synthetic_images.shape[0]} images)")

# -------------------------------
# Step 5: Merge Real & Synthetic Data for Training
# -------------------------------
print("🔄 Merging synthetic and real training data...")

# Load original training data
X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)

# Load synthetic Pneumothorax images
X_synthetic = np.load("X_synthetic.npy")
Y_synthetic = np.load("Y_synthetic.npy")

# Combine datasets
X_train_aug = np.concatenate([X_train, X_synthetic], axis=0)
Y_train_aug = np.concatenate([Y_train, Y_synthetic], axis=0)

# Save new dataset
np.save(OUTPUT_X_PATH, X_train_aug)
np.save(OUTPUT_Y_PATH, Y_train_aug)

print(f"✅ Final dataset: {X_train_aug.shape[0]} images (Real + Synthetic)")
