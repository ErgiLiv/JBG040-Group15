import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from dc1.net import Net
from dc1.image_dataset import ImageDataset
from dc1.batch_sampler import BatchSampler
from pathlib import Path

def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load the trained model weights and return the model.
    """
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model: nn.Module, batch_size: int = 100, device: str = "cpu"):
    """
    Run evaluation on the test set in batches to avoid memory overload.
    """

    # Load test dataset via ImageDataset to enable batching
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"))
    test_sampler = BatchSampler(batch_size=batch_size, dataset=test_dataset, balanced=False)

    all_labels = []
    all_probs = []
    all_predictions = []

    with torch.no_grad():
        for x_batch, y_batch in test_sampler:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()

            logits = model(x_batch).squeeze(1)
            probs = torch.sigmoid(logits)

            predicted_labels = (probs > 0.5).long()

            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)

    # Confusion Matrix
    conf_mat = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(conf_mat)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=["No Pneumothorax", "Pneumothorax"]))

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model_path = "model_weights/model_03_04_12_38.txt"  # Update with your trained model path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device)

    # Run evaluation in batches (batch size can be adjusted based on your system memory)
    evaluate_model(model, batch_size=100, device=device)
