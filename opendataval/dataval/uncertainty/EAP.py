# File modified -- this is a new file with our novel method

from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.uncertainty.rr_score import MANIFOLD
from scipy.stats import t
from scipy.stats import beta

class EAP(DataEvaluator, ModelMixin):
    def __init__(
            self, num_subsets: int = 100,
            k: int = 5,
            class_anom: int = 1,
            beta_c: np.array = None,
            random_state: Optional[int] = 331,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        ):
            self.num_subsets = num_subsets
            self.device = device
            self.k = k
            self.beta_c = beta_c
            self.class_anom = class_anom
            self.random_state = random_state
            torch.manual_seed(self.random_state)
            np.random.seed(random_state)
            self.evaluator_name = 'EAP'
            
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
        if self.beta_c is None:
            contamination = self.train_anomalies/ self.num_points
            self.beta_c = np.array([1-contamination, contamination], dtype=float)
        sum_beta = self.beta_c.sum()
        self.data_values = np.zeros(self.n_fakes, dtype=float)
        min_size = (self.k+1+self.train_anomalies)/self.num_points
        p_val = np.random.uniform(low=min_size, high=.99, size = self.num_subsets)
        
        if self.verbose:
            listvalues = tqdm(p_val)
        else:
            listvalues = p_val
            
        for p in listvalues:
            sel_idx = np.random.binomial(1, p, size=self.num_points)
            X_tr = self.x_train[np.where(sel_idx==1)[0],:].cpu().numpy()
            y_tr = self.y_train[np.where(sel_idx==1)[0]].cpu().numpy()
            curr_model = self.pred_model.clone()
            curr_model.fit(X_tr, y_tr, *args, **kwargs)
            class_prob = curr_model.predict(self.x_fakes.cpu()).cpu().numpy().reshape(-1)
            k_rarity = self.find_minimum_k(X_tr[y_tr != self.class_anom, :], self.x_train[self.y_train == self.class_anom, :].cpu().numpy(), min_k=self.k)
            manifold = MANIFOLD(real_features=X_tr[y_tr != self.class_anom, :],
                                fake_features=np.concatenate((X_tr[y_tr != self.class_anom, :], self.x_fakes.cpu().numpy())), device = self.device)
            rarity_score, _ = manifold.rarity(k=k_rarity)
            inverse_rarity_score = np.nan_to_num(1/rarity_score, neginf=0, posinf = 0)
            sum_real_inverse_rarity_scores = inverse_rarity_score[len(X_tr[np.where(y_tr!=self.class_anom)[0],:]):]+np.sum(inverse_rarity_score[:len(X_tr[np.where(y_tr!=self.class_anom)[0],:])])
            density = np.divide(inverse_rarity_score[len(X_tr[np.where(y_tr!=self.class_anom)[0],:]):], sum_real_inverse_rarity_scores)
            probs = np.divide(self.beta_c[self.class_anom] + len(X_tr[np.where(y_tr!=self.class_anom)[0],:])*class_prob*density, sum_beta+len(X_tr[np.where(y_tr!=self.class_anom)[0],:])*density)
            self.data_values += probs
        self.data_values = self.data_values/self.num_subsets
        return self
    
    def evaluate_data_values(self) -> np.array:
        return self.data_values
    
    
    def find_minimum_k_old(self, 
                       X1: np.array, 
                       X2: np.array, 
                       min_k: int = 5):
        k = min_k
        k_values = np.zeros(len(X2), int)

        while True:
            manifold = MANIFOLD(real_features=X1, fake_features=X2, device = self.device)
            scores, _ = manifold.rarity(k=k)
            k_values[(scores > 0) & (k_values == 0)] = k
            if np.prod(scores)>0:
                mu = np.median(k_values)
                sigma = np.std(k_values)
                if sigma>0:
                    min_stats_k = np.max(t.interval(0.95, len(X2), loc = mu, scale = sigma))
                    min_stats_k = int(np.nan_to_num(min_stats_k, neginf = len(X1)-1, posinf = len(X1)-1))
                else:
                    min_stats_k = min_k
                return min(len(X1) - 1, min_stats_k)
            k += 1
            if k >= len(X1):
                k_values[k_values == 0] = len(X1) - 1
                mu = np.median(k_values)
                sigma = np.std(k_values)
                if sigma>0:
                    min_stats_k = np.max(t.interval(0.95, len(X2), loc = mu, scale = sigma))
                    min_stats_k = int(np.nan_to_num(min_stats_k, neginf = len(X1)-1, posinf = len(X1)-1))
                else:
                    min_stats_k = min_k
                return min(len(X1) - 1, min_stats_k)        

    def find_minimum_k(self, 
                       X1: np.array, 
                       X2: np.array, 
                       min_k: int = 5):
        k = min_k
        k_values = np.zeros(len(X2), int)
        
        if len(X1)<= min_k+1:
            return len(X1) - 1

        while True:
            manifold = MANIFOLD(real_features=X1, fake_features=X2, device = self.device)
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