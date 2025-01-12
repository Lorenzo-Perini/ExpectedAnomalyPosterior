# File modified

from itertools import accumulate
from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.margcontrib.shap import ShapEvaluator


class DataBanzhaf(DataEvaluator, ModelMixin):

    def __init__(
        self, num_models: int = 100, 
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.num_models = num_models
        self.device = device
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.evaluator_name = 'DataBanzhaf'
        
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
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.sample_utility = torch.zeros((self.num_fakes_points, 2), dtype=torch.float, device=self.device)
        self.sample_counts = torch.zeros((self.num_fakes_points, 2), dtype=torch.float, device=self.device)

        return self

    def train_data_values(self, *args, **kwargs):

        sample_dim = (self.num_models, self.num_training_points + self.num_fakes_points)
        subsets = torch.zeros(sample_dim, dtype=torch.float, device=self.device).bernoulli_(0.5)
        
        if self.verbose:
            iterator = tqdm.tqdm(range(self.num_models))
        else:
            iterator = range(self.num_models)
        
        for i in iterator:
            try:
                subset = subsets[i].nonzero().view(-1)
                clean_subset = subset[subset<self.num_training_points]
                fake_subset = subset[subset>=self.num_training_points]
                if not clean_subset.any() or not clean_subset.any():
                    print("Entered inside the CONTINUE if clause")
                    continue
                clean_model = self.pred_model.clone()
                clean_model.fit(self.x_train[clean_subset,:].cpu(), self.y_train[clean_subset].cpu(), *args, **kwargs)
                y_train_hat = clean_model.predict(self.x_train)
                curr_perf = self.evaluate(self.y_train, y_train_hat, metric = self.metric)
                self.sample_utility[[x for x in range(self.num_fakes_points) if subsets[i][x + self.num_training_points] ==0], 0] += curr_perf
                self.sample_counts[[x for x in range(self.num_fakes_points) if subsets[i][x + self.num_training_points] ==0], 0] += 1
                #self.sample_utility[fake_subset, 0] += curr_perf
                #self.sample_counts[fake_subset, 0] += 1
                
                for j in fake_subset:
                    fake_model = self.pred_model.clone()
                    X_extended = torch.cat((self.x_train[clean_subset, :], self.x_fakes[j - self.num_training_points, :].reshape(1, -1)))
                    y_extended = torch.cat((self.y_train[clean_subset], self.y_fakes[j - self.num_training_points].reshape(-1))).reshape(-1)
                    fake_model.fit(X_extended.cpu(), y_extended.cpu(), *args, **kwargs)
                    y_train_hat = fake_model.predict(self.x_train)
                    curr_perf = self.evaluate(self.y_train, y_train_hat, metric = self.metric)

                    if j-self.num_training_points<0:
                        print("@@@@@@@@ If j - num_training_points is negative there is a problem! @@@@@@@@")
                    self.sample_utility[j - self.num_training_points, 1] += curr_perf
                    self.sample_counts[j - self.num_training_points, 1] += 1
            except:
                pass
        return self

    def evaluate_data_values(self) -> np.ndarray:
        msr = self.sample_utility / self.sample_counts
        msr[self.sample_counts == 0] = 0  # Handle division by zero
        data_values = msr[:, 1] - msr[:, 0]

        return data_values.cpu().numpy()  # Diff of subsets including/excluding i data point


class DataBanzhafMargContrib(ShapEvaluator):

    def compute_weight(self) -> float:

        def pascals(prev: int, position: int):  # Get level of pascal's triangle
            return (prev * (self.num_points - position + 1)) // position

        weights = torch.tensor(
            list(accumulate(range(1, self.num_points), func=pascals, initial=1)),
            dtype=torch.float,
            device=self.device
        )
        return weights / weights.sum()
