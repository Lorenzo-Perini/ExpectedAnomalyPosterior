# File modified

import math
from collections import Counter, defaultdict
from typing import Optional, Sequence
from numpy.random import RandomState
from sklearn.utils import check_random_state

import numpy as np
import torch

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.margcontrib import Sampler, TMCSampler


class RobustVolumeShapley(DataEvaluator, ModelLessMixin):
    """Robust Volume Shapley and Volume Shapley data valuation implementation.

    While the following DataEvaluator uses the same TMC-Shapley algorithm used by
    semivalue evaluators, the following implementation defaults to the non-GR statistic
    implementation. Instead a fixed number of samples is taken, which is
    closer to the original implementation here:
    https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/tree/main

    References
    ----------
    .. [1] X. Xu, Z. Wu, C. S. Foo, and B. Kian,
        Validation Free and Replication Robust Volume-based Data Valuation,
        Advances in Neural Information Processing Systems,
        vol. 34, pp. 10837-10848, Dec. 2021.

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.GrTMCSampler`.
    robust : bool, optional
        If the robust volume measure will be used which trades off a "more refined
        representation of diversity for greater robustness to replication",
        by default True
    omega : Optional[float], optional
        Width/discretization coefficient for x_train to be split into a set of d-cubes,
        required if `robust` is True, by default 0.05

    Mixins
    ------
    ModelLessMixin
        Mixin for a data evaluator that doesn't require a model or evaluation metric.
    """

    def __init__(
        self,
        sampler: Sampler = None,
        robust: bool = True,
        omega: Optional[float] = None,
        random_state: Optional[RandomState] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        *args,
        **kwargs,
    ):
        self.sampler = sampler
        self.robust = robust
        self.omega = omega if robust and omega is not None else 0.05
        self.device = device
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.evaluator_name = 'RobustVolumeShapley'
        
        if sampler is None:
            self.sampler = TMCSampler(random_state = self.random_state, device = self.device, *args, **kwargs)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_fakes: torch.Tensor,
        y_fakes: torch.Tensor,
        y_fake_true_label: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor
        ):
        self.x_train, self.x_fakes, self.x_test = self.embeddings(x_train, x_fakes, x_test)
        self.y_train, self.y_fakes, self.y_test = y_train, y_fakes, y_test
        self.y_fake_true_label = y_fake_true_label
        
        # Sampler parameters
        self.num_training_points = len(x_train)
        self.num_fakes_points = len(x_fakes)
        self.sampler.set_coalition(x_train, x_fakes, self.verbose)
        self.sampler.set_evaluator(self._evaluate_volume)
        return self

    def train_data_values(self, *args, **kwargs):
        self.marg_contrib = self.sampler.compute_marginal_contribution(*args, **kwargs)
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return np.sum(self.marg_contrib / self.num_fakes_points, axis=1)

    def _evaluate_volume(self, train_idx: list[int], fake_idx: int):
        
        if fake_idx > -1:
            X_extended = torch.cat((self.x_train[train_idx, :], self.x_fakes[fake_idx,:].reshape(1, -1)))
        else:
            X_extended = self.x_train[train_idx, :]
        
        if self.robust:
            x_tilde, cubes = compute_x_tilde_and_counts(X_extended, self.omega)
            return compute_robust_volumes(x_tilde, cubes)
        else:
            return torch.sqrt(torch.linalg.det(X_extended.T @ X_extended).abs() + 1e-8)


def compute_x_tilde_and_counts(x: torch.Tensor, omega: float):
    """Compresses the original feature matrix x to x_tilde with the specified omega.

    Returns
    -------
    np.ndarray
        Compressed form of x as a d-cube
    dict[tuple, int]
        A dictionary of cubes with the respective counts in each dcube
    """
    assert 0 <= omega <= 1.0, "`omega` must be in range [0, 1]"
    cubes = Counter()  # a dictionary to store the freqs
    omega_dict = defaultdict(list)
    min_ds = torch.min(x, axis=0).values

    # a dictionary to store cubes of not full size
    for entry in x:
        cube_key = tuple(math.floor(ent.item() / omega) for ent in entry - min_ds)
        cubes[cube_key] += 1
        omega_dict[cube_key].append(entry)

    x_tilde = torch.stack([torch.stack(value).mean(0) for value in omega_dict.values()])
    return x_tilde, cubes


def compute_robust_volumes(x_tilde: torch.Tensor, hypercubes: dict[tuple, int]):
    alpha = 1.0 / (10 * len(x_tilde))  # it means we set beta = 10

    flat_data = x_tilde.reshape(-1, x_tilde.shape[1])
    volume = torch.sqrt(torch.linalg.det(flat_data.T @ flat_data).abs() + 1e-8)
    rho_omega_prod = 1.0

    for freq_count in hypercubes.values():
        rho_omega_prod *= (1 - alpha ** (freq_count + 1)) / (1 - alpha)
    #print("Volume:", volume * rho_omega_prod)
    return volume * rho_omega_prod
