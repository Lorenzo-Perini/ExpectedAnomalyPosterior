# File modified

import numpy as np
from scipy.special import beta
import torch
from typing import Callable, ClassVar, Optional, TypeVar

from opendataval.dataval.margcontrib.shap import Sampler, ShapEvaluator

class BetaShapley(ShapEvaluator):
    """Beta Shapley implementation. Must specify alpha/beta values for beta function.

    References
    ----------
    .. [1] Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.GrTMCSampler`.
    alpha : int, optional
        Alpha parameter for beta distribution used in the weight function, by default 4
    beta : int, optional
        Beta parameter for beta distribution used in the weight function, by default 1
    """

    def __init__(
        self, sampler: Sampler = None, 
        alpha: int = 4, 
        beta: int = 1, 
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        *args, **kwargs
    ):
        super().__init__(sampler=sampler, device = device, *args, **kwargs)
        self.alpha, self.beta = alpha, beta  # Beta distribution parameters
        self.device = device
        self.random_state = random_state
        self.evaluator_name = 'BetaShapley'

    def compute_weight(self) -> torch.tensor:
        r"""Compute weights for each cardinality of training set.

        Uses :math:`\alpha`, :math:`beta` are parameters to the beta distribution.
        [1] BetaShap weight computation, :math:`j` is cardinality, Equation (3) and (5).

        .. math::
            w(j) := \frac{1}{n} w^{(n)}(j) \tbinom{n-1}{j-1}
            \propto \frac{Beta(j + \beta - 1, n - j + \alpha)}{Beta(\alpha, \beta)}
            \tbinom{n-1}{j-1}

        References
        ----------
        .. [1] Y. Kwon and J. Zou,
            Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
            Machine Learning,
            arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """
        num_points = self.sampler.num_training_points
        j_values = self.sampler.marginal_count.float()#.to(self.device)
        numerator = self.beta_function(j_values + self.beta, num_points - (j_values + 1) + self.alpha)
        denominator = self.beta_function(j_values + 1, num_points - j_values)
    
        weight_list = torch.div(numerator, denominator)
        
        return torch.div(weight_list, torch.sum(weight_list))

    def beta_function(self, alpha, beta):
        return torch.exp(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))