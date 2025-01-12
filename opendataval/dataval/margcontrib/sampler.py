# File modified

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional, TypeVar

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
import random

from opendataval.util import ReprMixin

Self = TypeVar("Self", bound="Sampler")


class Sampler(ABC, ReprMixin):
    """Abstract Sampler class for marginal contribution based data evaluators.

    Many marginal contribution based data evaluators depend on a sampling method as
    they typically can be very computationally expensive. The Sampler class provides
    a blue print of required methods to be used and the following samplers provide ways
    of caching computed marginal contributions if given a `"cache_name"`.
    """

    def set_evaluator(self, value_func: Callable[[list[int], ...], float]):
        """Sets the evaluator function to evaluate the utility of a coalition


        Parameters
        ----------
        value_func : Callable[[list[int], ...], float]
            This function sets the utility function  which computes the utility for a
            given coalition of indices.

        The following is an example of how the api would work in a DataEvaluator:
        ::
            self.sampler.set_evaluator(self._evaluate_model)
        """
        self.compute_utility = value_func

    @abstractmethod
    def set_coalition(self, train_coalition: torch.Tensor, fakes_coalition: torch.Tensor) -> Self:
        """Given the coalition, initializes data structures to compute marginal contrib.

        Parameters
        ----------
        coalition : torch.Tensor
            Coalition of data to compute the marginal contribution of each data point.
        """

    @abstractmethod
    def compute_marginal_contribution(self, *args, **kwargs) -> np.ndarray:
        """Given args and kwargs for the value func, computes marginal contribution.

        Returns
        -------
        np.ndarray
            Marginal contribution array per data point for each coalition size. Dim 0 is
            the index of the added data point, Dim 1 is the cardinality when the data
            point is added.
        """


class TMCSampler(Sampler):
    """TMCShapley sampler for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    mc_epochs : int, optional
        Number of outer epochs of MCMC sampling, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default "" which is set to a unique value for a object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 100,
        min_cardinality: int = 50,
        n_subsets : int = 100,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.n_subsets = n_subsets
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = device
        np.random.seed(random_state)

    def set_coalition(self, train_coalition: torch.Tensor, fakes_coalition: torch.Tensor, verbose: bool = False):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_training_points = len(train_coalition)
        self.num_fakes_points = len(fakes_coalition)
        self.verbose = verbose

        self.marginal_contrib_sum = np.zeros((self.num_fakes_points, self.num_training_points))
        self.marginal_count = np.zeros((self.num_fakes_points, self.num_training_points)) + 1e-8
        return self

    def compute_marginal_contribution(self, *args, **kwargs):
        """Computes marginal contribution through TMC Shapley.

        Uses TMC-Shapley sampling to find the marginal contribution of each data point,
        takes self.mc_epochs number of samples.
        """
        # Checks if data values have already been computed
        #if self.cache_name in self.CACHE:
        #    return self.CACHE[self.cache_name]        
        if self.verbose:
            iterator = tqdm.trange(self.mc_epochs)
        else:
            iterator = range(self.mc_epochs)

        for _ in iterator:
            self._calculate_marginal_contributions(*args, **kwargs)

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count

        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        subset = np.random.permutation(self.num_training_points)
        size_groups = np.random.randint(low = 1, high = max(min((2*self.num_training_points - self.min_cardinality)//self.n_subsets, self.num_training_points),2), size = 1)[0]
        
        marginal_increment = np.zeros(self.num_fakes_points) + 1e-8  # Prevents overflow
        ntimes = 0
        coalition = list(subset[: self.min_cardinality])
        # Baseline at minimal cardinality
        if self.verbose:
            iterator = tqdm.tqdm(np.arange(self.min_cardinality, self.num_training_points, size_groups))
        else:
            iterator = np.arange(self.min_cardinality, self.num_training_points, size_groups)
        for idx in iterator:
            # Increment the batch_size and evaluate the change compared to prev model
            coalition = subset[:idx]
            prev_perf = self.compute_utility(coalition, -1, *args, **kwargs)
            for fk_idx in range(self.num_fakes_points):
                curr_perf = self.compute_utility(coalition, fk_idx, *args, **kwargs)
                marginal_increment[fk_idx] = curr_perf - prev_perf
                # When the cardinality of random set is 'n',
                self.marginal_contrib_sum[fk_idx, idx] += curr_perf - prev_perf
                self.marginal_count[fk_idx, idx] += 1
                ntimes+=1
                
            # update performance
            self.marginal_count[np.where(self.marginal_count <1)] = 1
        return


class GrTMCSampler(Sampler):
    """TMC Sampler with terminator for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    .. [2]  Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
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

    GR_MAX = 100
    """Default maximum Gelman-Rubin statistic. Used for burn-in."""
    def __init__(
        self,
        gr_threshold: float = 1.05,
        max_mc_epochs: int = 10,
        models_per_epoch: int = 10,
        min_models: int = 20,
        min_cardinality: int = 50,
        n_subsets : int = 50,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        torch.manual_seed(random_state)
        self.max_mc_epochs = max_mc_epochs
        self.gr_threshold = gr_threshold
        self.models_per_epoch = models_per_epoch
        self.min_models = min_models
        self.min_cardinality = min_cardinality
        self.n_subsets = n_subsets
        self.random_state = random_state
        self.device = device
        np.random.seed(random_state)

    def set_coalition(self, train_coalition: torch.Tensor, fakes_coalition: torch.Tensor, verbose: bool = False):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_training_points = len(train_coalition)
        self.num_fakes_points = len(fakes_coalition)
        self.verbose = verbose

        self.marginal_contrib_sum = np.zeros((self.num_fakes_points, self.num_training_points))
        self.marginal_count = np.zeros((self.num_fakes_points, self.num_training_points)) + 1e-8

        # Used for computing the GR-statistic
        self.marginal_increment_array_stack = np.zeros((0, self.num_fakes_points))
        return self


    def compute_marginal_contribution(self, *args, **kwargs):
        """Compute the marginal contributions for semivalue based data evaluators.

        Computes the marginal contribution by sampling.
        Checks MCMC convergence every 100 iterations using Gelman-Rubin Statistic.
        NOTE if the marginal contribution has not been calculated, will look it up in
        a cache of already trained ShapEvaluators, otherwise will train from scratch.

        Parameters
        ----------
        args : tuple[Any], optional
             Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Notes
        -----
        marginal_increment_array_stack : np.ndarray
            Marginal increments when one data point is added.
        """

        print("Start: marginal contribution computation", flush=True)

        iteration = 0  # Iteration wise terminator, in case MCMC goes on for too long

        while iteration < self.max_mc_epochs:
            # we check the convergence every 100 random samples.
            # we terminate iteration if Shapley value is converged.
            samples_array = [self._calculate_marginal_contributions(*args, **kwargs) for _ in tqdm.tqdm(range(self.models_per_epoch))]
            self.marginal_increment_array_stack = np.vstack([self.marginal_increment_array_stack, *samples_array])

            iteration += 1  # Update terminating conditions

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count
        print("Done: marginal contribution computation", flush=True)

        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        subset = np.random.permutation(self.num_training_points)
        
        size_groups = np.random.randint(low = 1, high = max(min((2*self.num_training_points - self.min_cardinality)//self.n_subsets, self.num_training_points),2), size = 1)[0]
        #print("size_groups", size_groups)
        # for each iteration, we use random permutation for our MCMC
        
        marginal_increment = np.zeros(self.num_fakes_points) + 1e-8  # Prevents overflow
        avg_marginal_increment = np.zeros(self.num_fakes_points)+ 1e-8
        ntimes = 0
        coalition = list(subset[: self.min_cardinality])
        # Baseline at minimal cardinality
        if self.verbose:
            iterator = tqdm.tqdm(np.arange(self.min_cardinality, self.num_training_points, size_groups))
        else:
            iterator = np.arange(self.min_cardinality, self.num_training_points, size_groups)
        for idx in iterator:
            # Increment the batch_size and evaluate the change compared to prev model
            coalition = subset[:idx]
            prev_perf = self.compute_utility(coalition, -1, *args, **kwargs)
            for fk_idx in range(self.num_fakes_points):
                curr_perf = self.compute_utility(coalition, fk_idx, *args, **kwargs)
                marginal_increment[fk_idx] = curr_perf - prev_perf
                avg_marginal_increment[fk_idx] += curr_perf - prev_perf
                # When the cardinality of random set is 'n',
                self.marginal_contrib_sum[fk_idx, idx] += curr_perf - prev_perf
                self.marginal_count[fk_idx, idx] += 1
                ntimes+=1
                
            # update performance
            #prev_perf = curr_perf
            self.marginal_count[np.where(self.marginal_count == 0)] = 1
            avg_marginal_increment = avg_marginal_increment/ntimes
        return avg_marginal_increment.reshape(1, -1)

    