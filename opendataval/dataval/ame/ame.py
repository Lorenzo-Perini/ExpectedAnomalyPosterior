from typing import Optional
import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from scipy.stats import zscore
from sklearn.linear_model import LassoCV
from sklearn.utils import check_random_state
from torch.utils.data import Subset
from opendataval.dataval.api import DataEvaluator, ModelMixin


class AME(DataEvaluator, ModelMixin):

    def __init__(
        self, num_models: int = 100, 
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.num_models = num_models
        self.device = device
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.evaluator_name = 'AME'
        
    def train_data_values(self, *args, **kwargs):

        subsets, performance = [], []
        for proportion in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            sub, perf = (
                BaggingEvaluator(self.num_models, proportion, self.random_state, self.device, self.verbose)
                .input_model_metric(self.pred_model, self.metric)
                .input_data(self.x_train, self.y_train, self.x_fakes, self.y_fakes, self.y_fake_true_label, self.x_test, self.y_test)
                .train_data_values(*args, **kwargs)
                .get_subset_perf()
            )
            subsets.append(sub)
            performance.append(perf)
        self.performance = torch.stack(performance).view(-1)
        self.subsets = torch.stack(subsets).view(len(self.performance), -1)
        return self

    def evaluate_data_values(self) -> torch.Tensor:
        norm_subsets = (self.subsets - torch.mean(self.subsets, dim=1, keepdim=True)) / torch.std(self.subsets, dim=1, keepdim=True)
        norm_subsets[torch.isnan(norm_subsets)] = 0  # For when all elements are the same
        centered_perf = self.performance - torch.mean(self.performance)
        dv_ame = LassoCV(random_state=self.random_state)
        dv_ame.fit(X=norm_subsets.cpu().numpy(), y=centered_perf.cpu().numpy())
        return dv_ame.coef_[len(self.x_train):]


class BaggingEvaluator(DataEvaluator, ModelMixin):
    def __init__(
        self,
        num_models: int = 100,
        proportion: float = 1.0,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        verbose: bool = False,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.device = device
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.verbose = verbose

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
        return self

    def train_data_values(self, *args, **kwargs):
        sample_dim = (self.num_models, self.num_training_points+self.num_fakes_points)
        self.subsets = torch.bernoulli(self.proportion * torch.ones(sample_dim, device=self.device)).to(torch.float)
        self.performance = torch.zeros((self.num_models,), device=self.device)
        if self.verbose:
            iterator = tqdm.tqdm(range(self.num_models))
        else:
            iterator = range(self.num_models)

        for i in iterator:
            try:
                subset = self.subsets[i].nonzero(as_tuple=False).squeeze()
                subset = subset[subset < self.num_training_points]
                if not subset.any():
                    continue
                selected_fake = torch.randint(0, self.num_fakes_points, size=(1,), device=self.device).item()
                fake_subset = torch.zeros(self.num_fakes_points, dtype=torch.float, device=self.device)
                fake_subset[selected_fake] = 1
                curr_model = self.pred_model.clone()
                X_extended = torch.cat((self.x_train[subset, :], self.x_fakes[selected_fake, :].reshape(1, self.x_train.shape[1])))
                y_extended = torch.cat((self.y_train[subset], self.y_fakes[selected_fake].reshape(-1))).reshape(-1)

                curr_model.fit(X_extended.cpu(), y_extended.cpu(), *args, **kwargs)
                y_train_hat = curr_model.predict(self.x_train)
                curr_perf = self.evaluate(self.y_train, y_train_hat, metric = 'f1')
                self.performance[i] = curr_perf
                self.subsets[i, self.num_training_points:] = fake_subset
            except:
                pass
        return self

    def evaluate_data_values(self):
        norm_subsets = (self.subsets - torch.mean(self.subsets, dim=1, keepdim=True)) / torch.std(self.subsets, dim=1, keepdim=True)
        norm_subsets[torch.isnan(norm_subsets)] = 0
        centered_perf = self.performance - torch.mean(self.performance)

        dv_ame = LassoCV(random_state=self.random_state, tol = 0.001, max_iter=10000)
        dv_ame.fit(X=norm_subsets.cpu().numpy(), y=centered_perf.cpu().numpy())
        return dv_ame.coef_

    def get_subset_perf(self):
        return self.subsets, self.performance
