# File modified

from typing import Optional
import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.distributions.uniform import Uniform
from opendataval.dataval.api import DataEvaluator


class RandomEvaluator(DataEvaluator):
    """Completely Random DataEvaluator for baseline comparison purposes.

    Generates Random data values from Uniform[0.0, 1.0].

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, 
                 device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 random_state: Optional[int] = 331):
        #self.random_state = check_random_state(random_state)
        self.device = device
        torch.manual_seed(random_state)
        self.random_state = random_state
        self.evaluator_name = 'RandomEvaluator'

    def train_data_values(self, *args, **kwargs):
        """RandomEval does not train to find the data values."""
        pass

    def evaluate_data_values(self) -> np.ndarray:
        """Return random data values for each training data point."""
        uniform_distribution = Uniform(0.0, 1.0)
        random_values = uniform_distribution.sample((len(self.x_fakes),)).to(self.device)
        return random_values.cpu().numpy()
