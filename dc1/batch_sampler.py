import numpy as np
import torch
from dc1.image_dataset import ImageDataset
from typing import Generator, Tuple


class BatchSampler:
    def __init__(self, batch_size: int, dataset: ImageDataset, balanced: bool = False) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.balanced = balanced

        targets = self.dataset.targets

        if balanced:
            pneumo_indices = np.where(targets == 1)[0]
            non_pneumo_indices = np.where(targets == 0)[0]

            min_class_size = min(len(pneumo_indices), len(non_pneumo_indices))
            self.indexes = np.concatenate([
            np.random.choice(pneumo_indices, min(len(non_pneumo_indices), len(pneumo_indices) * 3), replace=True),
            non_pneumo_indices
            ])


        else:
            self.indexes = np.arange(len(dataset))

    def __len__(self) -> int:
        return (len(self.indexes) // self.batch_size) + 1

    def shuffle(self) -> None:
        np.random.shuffle(self.indexes)

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        self.shuffle()
        for i in range(0, len(self.indexes), self.batch_size):
            batch_indices = self.indexes[i:i+self.batch_size]
            images, labels = zip(*[self.dataset[idx] for idx in batch_indices])
            yield torch.stack(images), torch.tensor(labels)
