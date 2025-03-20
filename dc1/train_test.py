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
    model.train()  # sets model to training mode

    for batch in tqdm(train_sampler):
        x, y = batch
        x, y = x.to(device), y.to(device).float()  # converts target to float for BCE loss

        # forward pass
        predictions = model(x)

        # compute loss
        loss = loss_function(predictions, y)
        losses.append(loss.item())

        optimizer.zero_grad()  # resets gradients
        loss.backward()  # backpropagation
        optimizer.step()  # updates model weights

    return losses


def test_model(model, test_sampler, loss_function, device):
    model.eval()  # sets model to evaluation mode now
    losses = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_sampler:
            data, target = data.to(device), target.to(device).float()

            output = model(data)

            # convert logits to predictions using 0.5 threshold
            predicted = (torch.sigmoid(output) > 0.5).int()

            # compute loss
            loss = loss_function(output, target)
            losses.append(loss.item())

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    return losses, all_predictions, all_labels







