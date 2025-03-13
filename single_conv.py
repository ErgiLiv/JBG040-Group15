import numpy as np

# Load the original labels
Y_train = np.load("dc1/data/Y_train.npy")
Y_test = np.load("dc1/data/Y_test.npy")

# Convert to binary classification: 1 for class 5 (Pneumothorax), 0 otherwise
Y_train_binary = (Y_train == 5).astype(int)
Y_test_binary = (Y_test == 5).astype(int)

# Save the new binary labels
np.save("dc1/data/Y_train_binary.npy", Y_train_binary)
np.save("dc1/data/Y_test_binary.npy", Y_test_binary)
