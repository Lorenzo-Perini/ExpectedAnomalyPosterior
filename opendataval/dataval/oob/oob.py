# File modified

from collections import defaultdict
from typing import Optional
from torcheval.metrics.functional import auc
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin


class DataOob(DataEvaluator, ModelMixin):

    def __init__(
        self,
        num_models: int = 100,
        proportion: int = 1.0,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = device
        self.evaluator_name = 'DataOob'

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_fakes: torch.Tensor,
        y_fakes: torch.Tensor,
        y_fake_true_label: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,

        ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_fakes = x_fakes
        self.y_fakes = y_fakes
        self.y_fake_true_label = y_fake_true_label
        self.x_test = x_test
        self.y_test = y_test

        self.num_training_points = len(x_train)
        self.num_fakes_points = len(x_fakes)

        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape
        self.max_samples = round(self.proportion * self.num_training_points)

        self.oob_pred = torch.zeros((0, *self.label_dim), requires_grad=False, device = self.device)
        self.oob_indices = GroupingIndex()
        return self

    def train_data_values(self, *args, **kwargs):
        sample_dim = (self.num_models, self.max_samples)
        self.subsets = torch.randint(2, size=sample_dim, dtype=torch.float, device=self.device) * self.proportion
       
        if self.verbose:
            iterator = tqdm.tqdm(range(self.num_models))
        else:
            iterator = range(self.num_models)

        for i in iterator:
            try:
                in_bag = self.subsets[i]
                curr_model = self.pred_model.clone()
                curr_model.fit(self.x_train[in_bag.bool(), :].cpu(), self.y_train[in_bag.bool()].cpu(), *args, **kwargs)
                y_hat = curr_model.predict(self.x_fakes).to(self.device)
                self.oob_pred = torch.cat((self.oob_pred, y_hat), dim=0)
                self.oob_indices.add_indices(torch.arange(self.num_fakes_points, device=self.device).tolist())
            except:
                pass
        return self

    def evaluate_data_values(self) -> np.ndarray:
        self.data_values = torch.zeros(self.num_fakes_points, dtype=torch.float, device=self.device)
        for i, indices in self.oob_indices.items():
            oob_labels = self.y_fakes[i].expand((len(indices), *self.label_dim))
            self.data_values[i] = self.evaluate(oob_labels, self.oob_pred[indices], metric= self.metric)
        return self.data_values.cpu().numpy()


class GroupingIndex(defaultdict[int, list[int]]):
    def __init__(self, start: int = 0):
        super().__init__(list)
        self.position = start

    def add_indices(self, values: list[int]):
        for i in values:
            self.__getitem__(i).append(self.position)
            self.position += 1
