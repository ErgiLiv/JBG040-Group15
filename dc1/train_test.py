from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List

def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    losses = []
    model.train()  # Set model to training mode

    for batch in tqdm(train_sampler):
        x, y = batch
        x, y = x.to(device), y.to(device).float()  # Convert target to float for BCE loss

        # Forward pass
        predictions = model(x)  # No softmax here; CrossEntropyLoss expects raw logits

        # Compute loss
        loss = loss_function(predictions, y)  # Ensure y is long (integer class labels)
        losses.append(loss.item())

        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

    return losses


def test_model(model, test_sampler, loss_function, device):
    model.eval()  # Set model to evaluation mode
    losses = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_sampler:
            data, target = data.to(device), target.to(device).float()

            output = model(data)  # Raw logits

            # Convert logits to predictions using 0.5 threshold
            predicted = (torch.sigmoid(output) > 0.5).int()

            # Compute loss
            loss = loss_function(output, target)  # Ensure target is long (int labels)
            losses.append(loss.item())

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    return losses, all_predictions, all_labels







