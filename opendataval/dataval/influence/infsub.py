# File modified

from typing import Optional

import numpy as np
import torch
import tqdm

from opendataval.dataval.api import DataEvaluator, ModelMixin


class InfluenceSubsample(DataEvaluator, ModelMixin):
    """Influence computed through subsamples implementation.

    Compute influence of each training example on for the validation dataset
    through closely-related subsampled influence.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    samples : int, optional
        Number of models to fit to take to find data values, by default 1000
    proportion : float, optional
        Proportion of data points to be in each sample, cardinality of each subset is
        :math:`(p)(num_points)`, by default 0.7 as specified by V. Feldman and C. Zhang
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        num_models: int = 100,
        proportion: float = 0.7,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = device
        self.evaluator_name = 'InfluenceSubsample'
        #np.random.seed(331)
        
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

        self.num_training_points = len(x_train)
        self.num_fakes_points = len(x_fakes)
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.influence_matrix = torch.zeros((self.num_fakes_points, 2), dtype=torch.float, device = self.device)
        self.sample_counts = torch.zeros((self.num_fakes_points, 2), dtype=torch.float, device = self.device)
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the Influence Subsample Data Valuator by sampling from subsets of
        :math:`(p)(num_points)` cardinality and computing the performance with the
        :math:`i` data point and without the :math:`i` data point. The form of sampling
        is similar to the shapely value when :math:`p` is :math:`0.5: (V. Feldman).
        Likewise, if we sample not from the subsets of a specific cardinality but the
        uniform across all subsets, it is similar to the Banzhaf value.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        if self.verbose:
            iterator = tqdm.tqdm(range(self.num_models))
        else:
            iterator = range(self.num_models)
        for _ in iterator:
            # Random subset of cardinality `round(self.proportion * self.num_points)`
            subset = torch.randperm(self.num_training_points+self.num_fakes_points)[:round(self.proportion*(self.num_training_points+self.num_fakes_points))]
            clean_subset = subset[subset<self.num_training_points]
            fake_subset = subset[subset>=self.num_training_points]
            clean_model = self.pred_model.clone()
            clean_model.fit(self.x_train[clean_subset,:].cpu(), self.y_train[clean_subset].cpu(), *args, **kwargs)
            y_train_hat = clean_model.predict(self.x_train)
            curr_perf = self.evaluate(self.y_train, y_train_hat, metric = 'f1')
            self.influence_matrix[[x for x in range(self.num_fakes_points) if x + self.num_training_points not in subset], 0] += curr_perf
            self.sample_counts[[x for x in range(self.num_fakes_points) if x + self.num_training_points not in subset], 0] += 1
            
            for j in fake_subset:
                fake_model = self.pred_model.clone()
                X_extended = torch.cat((self.x_train[clean_subset, :], self.x_fakes[j-self.num_training_points,:].unsqueeze(0)))
                y_extended = torch.cat((self.y_train[clean_subset], self.y_fakes[j-self.num_training_points].unsqueeze(0))).view(-1)
                fake_model.fit(X_extended.cpu(), y_extended.cpu(), *args, **kwargs)
                y_train_hat = fake_model.predict(self.x_train)
                curr_perf = self.evaluate(self.y_train, y_train_hat, metric='f1')

                self.influence_matrix[j-self.num_training_points, 1] += curr_perf
                self.sample_counts[j-self.num_training_points, 1] += 1
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the Influence Subsample data valuator. Finds
        the difference of average performance of all sets including data point minus
        not-including.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = self.influence_matrix / self.sample_counts
        msr[self.sample_counts == 0] = 0
        data_values = msr[:, 1] - msr[:, 0]
        return data_values.cpu().numpy()
