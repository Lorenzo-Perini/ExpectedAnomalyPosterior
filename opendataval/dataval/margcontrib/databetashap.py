# File modified -- almost everything is different

from abc import ABC
from typing import Optional
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, accuracy_score

import numpy as np
import pandas as pd
import torch
from scipy.special import beta

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.margcontrib.sampler import GrTMCSampler, Sampler


class DataBetaShapEvaluator(DataEvaluator, ModelMixin, ABC):
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
                 alpha: int = 4, beta: int = 1, 
                 *args, **kwargs):
        self.sampler = sampler or GrTMCSampler(device = device, random_state = random_state, *args, **kwargs)
        self.device = device
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.alpha = alpha
        self.beta = beta
        #self.evaluator_name = 'DataShapley'

    def compute_weight_data(self) -> torch.tensor:
        """Compute the weights for each cardinality of training set."""
        self.evaluator_name = 'DataShapley'
        return np.array(1/self.num_training_points)

    def compute_weight_beta(self) -> torch.tensor:
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
        self.evaluator_name = 'BetaShapley'
        weight_list = [beta(j+self.beta,self.num_training_points-(j+1)+self.alpha)/beta(j+1,self.num_training_points-j) for j in range(self.num_training_points)]
        return np.array(weight_list) / np.sum(weight_list)

    
    def evaluate_data_values(self, which:str='data') -> np.ndarray:
        """Return data values for each training data point.

        Multiplies the marginal contribution with their respective weights to get
        data values for semivalue-based estimators

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every input data point
        """
        if which == 'data':
            weights = self.compute_weight_data()
        elif which == 'beta':
            weights = self.compute_weight_beta()
        return np.sum(self.marg_contrib * weights, axis=1)
    
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
        self.marg_contrib = self.sampler.compute_marginal_contribution(self.verbose, *args, **kwargs)#.cpu().numpy()
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
            y_extended = self.y_train[train_idx].reshape(-1)
        curr_model.fit(X_extended.cpu().numpy(), y_extended.cpu().numpy(), *args, **kwargs )
        y_train_hat = curr_model.predict(self.x_train)

        curr_perf = self.evaluate(self.y_train, y_train_hat, metric = self.metric)
        return curr_perf
    
    #def beta_function(self, alpha, beta):
    #    return torch.exp(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
    
    
    def evaluate_quality_metric(self, model: str = 'svm', metric: str = 'accuracy', path: str = '', save_LC: bool = True, save_OP: bool = True, save_DV: bool = True,):

        y_fakes_good = np.copy(self.y_fake_true_label.cpu().numpy())
        y_fakes_good[y_fakes_good!= 1] = 0
        
        for which in ['data', 'beta']:

            if len(path) > 0:
                self.df_learningcurve = pd.read_csv(path)
            else:
                self.compute_learning_curve(model = model, metric = metric, save_LC = save_LC, save_DV = save_DV, which = which)
            
            overall_performance = [self.evaluator_name]    

            ##### AREA UNDER THE LEARNING CURVE #####
            #Max - No contamination
            overall_performance.append(np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_max'].values, dx = 1/len(self.df_learningcurve)))
            #Max - True contamination
            overall_performance.append(np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_max'].values[:len(np.where(y_fakes_good==1)[0])],
                                                dx = 1/len(np.where(y_fakes_good==1)[0])))
            #Min - No contamination
            overall_performance.append(-np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_min'].values, dx = 1/len(self.df_learningcurve)))
            #Min - True contamination
            overall_performance.append(-np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_min'].values[:len(np.where(y_fakes_good==0)[0])],
                                                dx = 1/len(np.where(y_fakes_good==0)[0])))

            ##### SSDO'S F1 SCORE ADDING FAKES #####
            
            threshold_range = np.linspace(np.min(self.data_values_), np.max(self.data_values_), num=100)
            criterias = [compute_otsu_criteria(self.data_values_, th) for th in threshold_range]

            # best threshold is the one minimizing the Otsu criteria
            best_threshold = threshold_range[np.argmin(criterias)]
            clstrs = np.zeros(len(self.data_values_), int)
            clstrs[np.where(self.data_values_>best_threshold)[0]] = 1

            good_points = np.argmax([self.data_values_[np.where(clstrs == 0)[0]].mean(),self.data_values_[np.where(clstrs == 1)[0]].mean()])
            sortlist = np.argsort(-1*self.data_values_.reshape(-1))[:len(np.where(y_fakes_good==1)[0])]

            ##### F1 SCORE OF QUALITY FAKE RECOGNITION #####
            # Using Otsu to create the clusters
            y_fakes_good_predicted = np.zeros(len(y_fakes_good), int)
            y_fakes_good_predicted[np.where(clstrs == good_points)[0]] = 1
            overall_performance.append(accuracy_score(y_true=y_fakes_good, y_pred=y_fakes_good_predicted))
            # Using the True Contamination
            y_fakes_good_predicted = np.zeros(len(y_fakes_good), int)
            y_fakes_good_predicted[sortlist] = 1
            overall_performance.append(accuracy_score(y_true=y_fakes_good, y_pred=y_fakes_good_predicted))
            
            ##### Q1. Do the methods clearly separate good fakes from polluted (i.e., normal) fakes? #####
            #split_polluted_fakes = roc_auc_score(y_true=y_fakes_polluted[np.where(y_noise==0)[0]], y_score = df[col].values[np.where(y_noise==0)[0]])
            #split_polluted_fakes_list.append(split_polluted_fakes)
            
            ##### Q2. Do the methods clearly separate good fakes from noise? #####
            #split_noise = roc_auc_score(y_true=y_fakes_good[np.where(y_fakes_polluted!=-1)[0]], y_score = df[col].values[np.where(y_fakes_polluted!=-1)[0]])
            #split_noise_list.append(split_noise)
            
            ##### Q3. Do the methods clearly separate good fakes from noise and polluted? #####
            overall_performance.append(roc_auc_score(y_true=y_fakes_good, y_score = self.data_values_))
            overall_performance.append(self.cpu_time)
            
            overall_performance.append(self.random_state)
            self.df_overall_performance = pd.DataFrame(data = [overall_performance], columns = ['evaluator', 'aulc_max','aulc_max_TrueCont','aulc_min',
                                                                                                'aulc_min_TrueCont', 'Acc_FakeRecog', 'Acc_FakeRecog_TrueCont',
                                                                                                'split_fakes', 'cpu_time', 'seed'])
            if save_OP:
                self.save_results_eval(self.df_overall_performance, path = "/home/pel2rng/Desktop/python/Results/csvfiles/", 
                                       name=model+"_"+self.dataset_name+"_"+metric+"_Evaluation")
        return self

    def compute_learning_curve(self, model: str = 'svm', metric: str = 'accuracy', save_LC: bool = True, save_DV: bool = True, which: str = 'data') -> pd.DataFrame:
        self.df_learningcurve['x_values'] = np.arange(0, len(self.x_fakes)+1)
        
        randomidxs = np.arange(0, len(self.x_fakes))
        np.random.shuffle(randomidxs)
        self.x_fakes_sorted = self.x_fakes[randomidxs, :].cpu().numpy()
        self.data_values_ = self.evaluate_data_values(which = which)
        if save_DV:
            self.save_results_dv(self.data_values_, path = "/home/pel2rng/Desktop/python/Results/csvfiles/", 
                                 name=self.dataset_name+"_"+str(self.random_state)+"_DataValues")
            
        print("Computing the learning curve following the metric's ordering ...")
        ranks = pd.DataFrame(self.data_values_[randomidxs]).rank(method = 'first', ascending = False).values
        self.df_learningcurve[self.evaluator_name+'_lc_max'] = self.evaluate_model_follow_ranking(ranks = ranks, model = model, metric = metric)

        print("Computing the learning curve following the metric's INVERSE ordering ...")
        ranks = pd.DataFrame(self.data_values_[randomidxs]).rank(method = 'first', ascending = True).values
        self.df_learningcurve[self.evaluator_name+'_lc_min'] = self.evaluate_model_follow_ranking(ranks = ranks, model = model, metric = metric)
        
        if save_LC:
            self.save_results_lc(df = self.df_learningcurve, path = "/home/pel2rng/Desktop/python/Results/csvfiles/",
                                 name = model+"_"+self.dataset_name+"_"+metric+"_"+str(self.random_state)+"_LearningCurve")
        return



def compute_otsu_criteria(im, th):
    """Otsu's method to compute criteria."""
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1