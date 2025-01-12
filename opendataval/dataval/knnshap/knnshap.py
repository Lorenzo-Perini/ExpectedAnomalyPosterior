# File modified

from typing import Optional

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class KNNShapley(DataEvaluator, ModelLessMixin):

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Optional[Model] = None,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = device
        self.evaluator_name = 'KNNShapley'

    def match(self, y: torch.Tensor) -> torch.Tensor:
        return (y == self.y_train).all(dim=0).float()

    def train_data_values(self, *args, **kwargs):
        n = len(self.x_train)
        n_fakes = len(self.x_fakes)
        x_train, x_fakes = self.embeddings(self.x_train, self.x_fakes)

        # Computes Euclidean distance by computing crosswise per batch
        # Doesn't shuffle to maintain relative order
        self.data_values = torch.zeros(n_fakes, dtype=torch.float, device = self.device)
        if self.verbose:
            iterator = tqdm.tqdm(range(n_fakes))
        else:
            iterator = range(n_fakes)
        for j in iterator:
            x_train_extended = torch.cat((x_train, x_fakes[j, :].reshape(1, -1)))
            y_train_extended = torch.cat((self.y_train, self.y_fakes[j].reshape(-1)))

            x_train_extended_view, x_train_view = x_train_extended.view(n + 1, -1), x_train.view(n, -1)

            dist_list = []  # Uses batching to only load at most `batch_size` tensors
            for x_train_batch in DataLoader(x_train_extended_view, batch_size=self.batch_size):  # No shuffle
                dist_row = []
                for x_val_batch in DataLoader(x_train_view, batch_size=self.batch_size):
                    dist_row.append(torch.cdist(x_train_batch, x_val_batch))
                dist_list.append(torch.cat(dist_row, dim=1))
            dist = torch.cat(dist_list, dim=0)

            # Arranges by distances
            sort_indices = torch.argsort(dist, dim=0, stable=True)
            y_train_sort = y_train_extended[sort_indices]

            score = torch.zeros_like(dist, device = self.device)
            score[sort_indices[n - 1], range(n)] = self.match(y_train_sort[n - 1]) / n

            for i in range(n - 2, -1, -1):
                score[sort_indices[i], range(n)] = (
                    score[sort_indices[i + 1], range(n)]
                    + min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1))
                    * (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
                )
            self.data_values[j] = score.mean(dim=1)[-1]

        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self.data_values.cpu().numpy()
