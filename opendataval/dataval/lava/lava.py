# File modified

from typing import Optional

import numpy as np
import torch
import tqdm

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.lava.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model

class LavaEvaluator(DataEvaluator, ModelLessMixin):
    def __init__(
        self,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        embedding_model: Optional[Model] = None,
        random_state: Optional[int] = 331,
    ):
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = device
        self.embedding_model = embedding_model
        self.evaluator_name = 'LavaEvaluator'
        
    def train_data_values(self, *args, **kwargs):
        feature_cost = None

        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            resize = 32
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=(3, resize, resize),
                tgt_embedding=self.embedding_model,
                tgt_dim=(3, resize, resize),
                p=2,
                device=self.device.type,
            )
            
        self.dual_sol = {}
        
        if self.verbose:
            iterator = tqdm.tqdm(range(len(self.x_fakes)))
        else:
            iterator = range(len(self.x_fakes))
            
        for j in iterator:
            x_extended = torch.cat((self.x_train, self.x_fakes[j,:].reshape(1,-1)))
            y_extended = torch.cat((self.y_train, self.y_fakes[j].reshape(-1)))
            x_extended, y_extended = self.embeddings(x_extended, y_extended)
            dist = DatasetDistance(x_train=x_extended, y_train=y_extended, x_valid=self.x_train, y_valid=self.y_train,
                                   feature_cost=feature_cost if feature_cost else "euclidean", lam_x=1.0, lam_y=1.0, p=2, entreg=1e-1, device=self.device)
            self.dual_sol[j] = dist.dual_sol()
        return self

    def evaluate_data_values(self) -> np.array:
        data_values = torch.zeros(len(self.x_fakes), dtype=torch.float, device=self.device)
        for j in range(len(self.x_fakes)):
            f1k = self.dual_sol[j][0].squeeze()
            num_points = len(f1k) - 1
            train_gradient = f1k * (1 + 1 / (num_points)) - f1k.sum() / num_points
            train_gradient = -1 * train_gradient
            data_values[j] = train_gradient[-1]
        
        return data_values.cpu().numpy()

