# File modified

from typing import Optional
import numpy as np
import torch
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.uncertainty.rr_score import MANIFOLD

class rarity_score(DataEvaluator, ModelMixin):
    def __init__(
            self,
            k: int = 10, 
            random_state: Optional[int] = 331,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        ):
            self.device = device
            self.k = k
            self.random_state = random_state
            torch.manual_seed(self.random_state)
            np.random.seed(random_state)
            self.evaluator_name = 'RarityScore'
            
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
        self.n_fakes = len(y_fakes)
        self.num_points = len(x_train)
        return self
    
    def train_data_values(self, *args, **kwargs):
        manifold = MANIFOLD(real_features=self.x_train.cpu().numpy(), fake_features=self.x_fakes.cpu().numpy(), device = self.device)
        self.data_values, _ = manifold.rarity(k=self.k)
        print()
        return self

    def evaluate_data_values(self) -> np.array:
        return self.data_values
    
