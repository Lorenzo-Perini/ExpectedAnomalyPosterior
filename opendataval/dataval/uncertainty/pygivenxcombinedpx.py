# File modified -- this is a new file with our novel method

from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.uncertainty.rr_score import MANIFOLD
from scipy.stats import t
from scipy.stats import beta

class PyGivenXandPXEstimator(DataEvaluator, ModelMixin):
    def __init__(
            self,
            k: int = 5,
            class_anom: int = 1,
            random_state: Optional[int] = 331,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        ):
            self.device = device
            self.k = k
            self.class_anom = class_anom
            self.random_state = random_state
            torch.manual_seed(self.random_state)
            np.random.seed(random_state)
            self.evaluator_name = 'CondProbDensity'
            
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

        X_tr = self.x_train.cpu().numpy()
        y_tr = self.y_train.cpu().numpy()
        curr_model = self.pred_model.clone()
        curr_model.fit(X_tr, y_tr, *args, **kwargs)
        class_prob = curr_model.predict(self.x_fakes.cpu()).cpu().numpy().reshape(-1)
        k_rarity = self.find_minimum_k(X_tr[y_tr != self.class_anom, :], self.x_train[self.y_train == self.class_anom, :].cpu().numpy(), min_k=self.k)
        manifold = MANIFOLD(real_features=X_tr[y_tr != self.class_anom, :],
                            fake_features=np.concatenate((X_tr[y_tr != self.class_anom, :], self.x_fakes.cpu().numpy())), device = "cpu")
        rarity_score, _ = manifold.rarity(k=k_rarity)
        inverse_rarity_score = np.nan_to_num(1/rarity_score, neginf=0, posinf = 0)
        sum_real_inverse_rarity_scores = inverse_rarity_score[len(X_tr[np.where(y_tr!=self.class_anom)[0],:]):]+np.sum(inverse_rarity_score[:len(X_tr[np.where(y_tr!=self.class_anom)[0],:])])
        density = np.divide(inverse_rarity_score[len(X_tr[np.where(y_tr!=self.class_anom)[0],:]):], sum_real_inverse_rarity_scores)
        self.data_values = len(X_tr)*density + class_prob
        return self
    
    def evaluate_data_values(self) -> np.array:
        return self.data_values
    
    
    def find_minimum_k(self, 
                       X1: np.array, 
                       X2: np.array, 
                       min_k: int = 5):
        k = min_k
        k_values = np.zeros(len(X2), int)
        
        if len(X1)<= min_k+1:
            return len(X1) - 1

        while True:
            manifold = MANIFOLD(real_features=X1, fake_features=X2, device = "cpu")
            scores, _ = manifold.rarity(k=k)
            k_values[(scores > 0) & (k_values == 0)] = k
            if np.prod(scores)>0:
                pstar = (k_values-min_k)/(len(X1)-1 -min_k)
                a = 1+ np.sum(pstar)
                b = 1+len(pstar) - np.sum(pstar)
                kstar = int(beta.ppf(0.95, a, b)*(len(X1)-1-min_k) + min_k)
                return max(min(len(X1) - 1, kstar),min_k)
            k += 1
            if k >= len(X1):
                k_values[k_values == 0] = len(X1) - 1
                pstar = (k_values-min_k)/(len(X1)-1 -min_k)
                a = 1+ np.sum(pstar)
                b = 1+len(pstar) - np.sum(pstar)
                kstar = int(beta.ppf(0.95, a, b)*(len(X1)-1-min_k) + min_k)
                return max(min(len(X1) - 1, kstar),min_k)