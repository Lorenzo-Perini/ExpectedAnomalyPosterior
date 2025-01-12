# File modified

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional, TypeVar
from sklearn.utils import check_random_state

import numpy as np
import torch
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.margcontrib.sampler import GrTMCSampler, Sampler


class ShapEvaluator(DataEvaluator, ModelMixin, ABC):
    """Abstract class for all semivalue-based methods of computing data values.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    .. [2]  Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Attributes
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contribution, by default uses
        TMC-Shapley with a Gelman-Rubin statistic terminator. Samplers are found in
        :py:mod:`~opendataval.margcontrib.sampler`

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        opendataval/margcontrib/sampler.py, by default GrTMCSampler and uses additonal
        arguments as constructor for sampler.
    gr_threshold : float, optional
        Convergence threshold for the Gelman-Rubin statistic.
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.05
    max_mc_epochs : int, optional
        Max number of outer epochs of MCMC sampling, by default 100
    models_per_epoch : int, optional
        Number of model fittings to take per epoch prior to checking GR convergence,
        by default 100
    min_models : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default "" which is set to a unique value for a object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, sampler: Sampler = None,
                 random_state: Optional[int] = 331,
                 device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 *args, **kwargs):
        self.sampler = sampler or GrTMCSampler(device = device, random_state = random_state, *args, **kwargs)
        self.device = device
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.evaluator_name = 'DataShapley'
    @abstractmethod
    def compute_weight(self) -> np.ndarray:
        """Compute the weights for each cardinality of training set."""

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Multiplies the marginal contribution with their respective weights to get
        data values for semivalue-based estimators

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every input data point
        """
        weights = self.compute_weight()
        weighted_data_values = torch.tensor([torch.sum(self.marg_contrib[i]*weights) for i in range(self.sampler.num_fakes_points)])
        return weighted_data_values.numpy()
    
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

        # Sampler specific setup
        self.num_training_points = len(x_train)
        self.sampler.set_coalition(x_train, x_fakes, self.verbose)
        self.sampler.set_evaluator(self._evaluate_model)

        return self

    def train_data_values(self, *args, **kwargs):
        """Uses sampler to trains model to find marginal contribs and data values."""
        self.marg_contrib = self.sampler.compute_marginal_contribution(self.verbose, *args, **kwargs)
        return self

    def _evaluate_model(self, train_idx: list[int], fake_idx: int, *args, **kwargs):
        """Evaluate performance of the model on a subset of the training data set.

        Parameters
        ----------
        subset : list[int]
            indices of covariates/label to be used in training
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        float
            Performance of subset of training data set
        """
        curr_model = self.pred_model.clone()
        if fake_idx > -1:
            X_extended = torch.cat((self.x_train[train_idx, :], self.x_fakes[fake_idx,:].reshape(1, self.x_train.shape[1])))
            y_extended = torch.cat((self.y_train[train_idx], self.y_fakes[fake_idx].reshape(-1))).reshape(-1)
        else:
            X_extended = self.x_train[train_idx, :]
            y_extended = self.y_train[train_idx]
        curr_model.fit(X_extended.cpu(), y_extended.cpu(), *args, **kwargs )
        y_train_hat = curr_model.predict(self.x_train)

        curr_perf = self.evaluate(self.y_train.cpu(), y_train_hat.cpu(), metric = self.metric)
        return curr_perf
