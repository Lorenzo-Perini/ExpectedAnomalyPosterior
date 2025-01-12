from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from opendataval.dataval.api import DataEvaluator, ModelMixin

class PyGivenXEstimator(DataEvaluator, ModelMixin):
    def __init__(
            self,
            class_anom: int = 1,
            random_state: Optional[int] = 331,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        ):
            self.device = device
            self.class_anom = class_anom
            self.random_state = random_state
            torch.manual_seed(self.random_state)
            np.random.seed(random_state)
            self.evaluator_name = 'CondProb'
            
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
        self.train_anomalies = len(torch.where(self.y_train == self.class_anom)[0])
        return self
    
    def train_data_values(self, *args, **kwargs):
        self.data_values = np.zeros(self.n_fakes, dtype=float)
        curr_model = self.pred_model.clone()
        curr_model.fit(self.x_train, self.y_train, *args, **kwargs)
        class_prob = curr_model.predict(self.x_fakes.cpu()).cpu().numpy().reshape(-1)
        self.data_values = class_prob
        return self
    
    def evaluate_data_values(self) -> np.array:
        return self.data_values
    
    
