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
        x, y = x.to(device), y.to(device)  # Move data to the correct device

        # Forward pass
        predictions = model(x)  # No softmax here; CrossEntropyLoss expects raw logits

        # Compute loss
        loss = loss_function(predictions, y.long())  # Ensure y is long (integer class labels)
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
    all_probs = []

    with torch.no_grad():
        for data, target in test_sampler:
            data, target = data.to(device), target.to(device)

            output = model(data)  # Raw logits

            # Convert logits to probabilities using softmax
            probs = torch.softmax(output, dim=1)

            # Get predicted class (argmax)
            predicted = torch.argmax(probs, dim=1)

            # Compute loss
            loss = loss_function(output, target.long())  # Ensure target is long (int labels)
            losses.append(loss.item())

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # Store class probabilities

    return losses, all_predictions, all_labels, all_probs







