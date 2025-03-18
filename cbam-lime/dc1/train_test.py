from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List


def train_model(model: Net, train_sampler: BatchSampler, optimizer: torch.optim.Optimizer,
                loss_function: Callable[..., torch.Tensor], device: str) -> List[torch.Tensor]:
    losses = []
    model.train()

    for batch in tqdm(train_sampler):
        x, y = batch
        x, y = x.to(device), y.to(device).float()

        predictions = model(x).squeeze(1)
        loss = loss_function(predictions, y)

        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def test_model(model: Net, test_sampler: BatchSampler, loss_function: Callable[..., torch.Tensor], device: str) -> List[torch.Tensor]:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(test_sampler):
            x, y = batch
            x, y = x.to(device), y.to(device).float()

            predictions = model(x).squeeze(1)
            loss = loss_function(predictions, y)

            losses.append(loss)

    return losses
