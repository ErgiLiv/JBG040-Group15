import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from pathlib import Path
from dc1.image_dataset import ImageDataset
from dc1.batch_sampler import BatchSampler


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load the trained ResNet model weights and return the model.
    """
    # Create ResNet model with same architecture as training
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Flatten(0, 1)
    )
    
    # Load weights and set to eval mode
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model: nn.Module, batch_size: int = 100, device: str = "cpu"):
    """
    Run evaluation on the test set in batches to avoid memory overload.
    """
    # Load test dataset
    test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test_binary.npy"))
    test_sampler = BatchSampler(batch_size=batch_size, dataset=test_dataset, balanced=False)

    all_labels = []
    all_probs = []
    all_predictions = []

    with torch.no_grad():
        for x_batch, y_batch in test_sampler:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
            
            # Get model predictions
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            predicted_labels = (probs > 0.5).long()

            # Store results
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=["Non-Pneumothorax", "Pneumothorax"]))

    # Calculate and print ROC metrics
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nAUROC Score: {auc:.3f}")

    # Calculate and print PR metrics
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    print(f"Average Precision Score: {avg_precision:.3f}")

    # Save plots to artifacts directory
    Path("artifacts/").mkdir(exist_ok=True)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path("artifacts") / f"confusion_matrix_evaluation.png")
    plt.close()

    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(Path("artifacts") / f"roc_curve_evaluation.png")
    plt.close()

    # Plot and save PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(Path("artifacts") / f"pr_curve_evaluation.png")
    plt.close()

if __name__ == "__main__":
    # Specify model path directly
    model_path = "model_weights/resnet18_model_03_13_02_09.pt"  # Update with your model path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and evaluate model
    model = load_model(model_path, device)
    evaluate_model(model, batch_size=100, device=device)
