# File modified

import numpy as np
import torch
import tqdm
from torch.utils.data import Subset
from typing import Optional
from opendataval.dataval.api import DataEvaluator, ModelMixin
from collections import Counter

class LeaveOneOut(DataEvaluator, ModelMixin):
    """Leave One Out data valuation implementation.

    References
    ----------
    .. [1] R. Cook,
        Detection of Influential Observation in Linear Regression,
        Technometrics, Vol. 19, No. 1 (Feb., 1977), pp. 15-18 (4 pages).

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
        self.evaluator_name = 'LeaveOneOut'
        
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

        # Additional parameters
        self.num_points = len(x_fakes)

        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Compute the data values using the Leave-One-Out data valuation.
        Equivalently, LOO can be computed from the marginal contributions as it's a
        semivalue.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        self.evaluator_name = 'LeaveOneOut'
        self.data_values = torch.zeros((self.num_points,), device = self.device)

        curr_model = self.pred_model.clone()
        curr_model.fit(self.x_train.cpu(), self.y_train.cpu(), *args, **kwargs)
        y_train_hat = curr_model.predict(self.x_train)
        baseline_score = self.evaluate(self.y_train, y_train_hat, metric = self.metric)
        if self.verbose:
            iterator = tqdm.tqdm(range(self.num_points))
        else:
            iterator = range(self.num_points)

        for i in iterator:
            curr_model = self.pred_model.clone()
            X_extended = torch.cat((self.x_train, self.x_fakes[i, :].unsqueeze(0)))
            y_extended = torch.cat((self.y_train, self.y_fakes[i].unsqueeze(0))).view(-1)
            curr_model.fit(X_extended.cpu(), y_extended.cpu(), *args, **kwargs)
            y_hat = curr_model.predict(self.x_train)
            loo_score = self.evaluate(self.y_train, y_hat, metric= self.metric)
            self.data_values[i] = loo_score - baseline_score

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Compute data values using Leave One Out data valuation.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values.cpu().numpy()
