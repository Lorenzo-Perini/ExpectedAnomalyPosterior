# File modified

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
import pandas as pd

from opendataval.dataloader import DataFetcher
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, accuracy_score
from opendataval.model import Model
from opendataval.util import ReprMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from datetime import datetime
from sklearn.base import clone
from tqdm import tqdm
import time

Self = TypeVar("Self", bound="DataEvaluator")


class DataEvaluator(ABC, ReprMixin):
    """Abstract class of Data Evaluators. Facilitates Data Evaluation computation.

    The following is an example of how the api would work:
    ::
        dataval = (
            DataEvaluator(*args, **kwargs)
            .input_data(x_train, y_train, x_valid, y_valid)
            .train_data_values(batch_size, epochs)
            .evaluate_data_values()
        )

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    args : tuple[Any]
        DavaEvaluator positional arguments
    kwargs : Dict[str, Any]
        DavaEvaluator key word arguments

    Attributes
    ----------
    pred_model : Model
        Prediction model to find how much each training datum contributes towards it.
    data_values: np.array
        Cached data values, used by :py:mod:`opendataval.experiment.exper_methods`
    """

    Evaluators: ClassVar[dict[str, Self]] = {}

    def __init__(self, random_state: Optional[int] = None, 
                 device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 verbose: bool = True,
                 path_save_results: str = '',
                 *args, **kwargs):
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.device = device
        self.evaluator_name = ''
        self.verbose = verbose
        self.path_save_results = path_save_results

        
    def __init_subclass__(cls, *args, **kwargs):
        """Registers DataEvaluator types, used as part of the CLI."""
        super().__init_subclass__(*args, **kwargs)
        cls.Evaluators[cls.__name__.lower()] = cls

    def input_data(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: torch.Tensor,
        #x_valid: Union[torch.Tensor, Dataset],
        #y_valid: torch.Tensor,
        x_fakes: Union[torch.Tensor, Dataset],
        y_fakes: torch.Tensor,
        y_fake_true_label: torch.Tensor,
        x_test: Union[torch.Tensor, Dataset],
        y_test: torch.Tensor,

    ):
        """Store and transform input data for DataEvaluator.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_fakes = x_fakes
        self.y_fakes = y_fakes
        self.y_fake_true_label = y_fake_true_label
        self.x_test = x_test
        self.y_test = y_test

        return self

    def setup(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[str] = 'f1',
        verbose: bool = False,
    ):
        """Inputs model, metric and data into Data Evaluator.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.verbose = verbose
        self.input_fetcher(fetcher)
        self.metric = metric
        self.df_learningcurve = pd.DataFrame(data = [], columns = [])
        self.cpu_time = 0.0
        self.formatted_date = datetime.now().strftime("%y%m%d")
        
        self.input_model(pred_model).input_metric(metric)
        return self

    def train(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[str] = 'f1',
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """Store and transform data, then train model to predict data values.

        Trains the Data Evaluator and the underlying prediction model. Wrapper for
        ``self.input_data`` and ``self.train_data_values`` under one method.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        current_date = datetime.now()
        self.formatted_date = current_date.strftime("%y%m%d")
        self.setup(fetcher, pred_model, metric, verbose)
        self.start_time = time.process_time()
        self.train_data_values(*args, **kwargs)
        self.end_time = time.process_time()
        self.cpu_time = self.end_time - self.start_time
        return self

    @abstractmethod
    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a trained Data Evaluator.
        """
        return self

    @abstractmethod
    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """

    @cached_property
    def data_values(self) -> np.ndarray:
        """Cached data values."""
        return self.evaluate_data_values()

    def input_fetcher(self, fetcher: DataFetcher):
        """Input data from a DataFetcher object. Alternative way of adding data."""
        x_train, y_train, x_fakes, y_fakes, y_fake_true_label, x_test, y_test = fetcher.datapoints()
        
        return self.input_data(x_train, y_train, x_fakes, y_fakes, y_fake_true_label, x_test, y_test)
    
    def evaluate_quality_metric(self, model: str = 'svm', metric: str = 'accuracy', lc_path: str = '', dv_path: str = '', 
                                save_LC: bool = True, save_OP: bool = True, save_DV: bool = True):
                
        y_fakes_good = np.copy(self.y_fake_true_label.cpu().numpy())
        y_fakes_good[y_fakes_good!= 1] = 0

        if len(lc_path) > 0:
            self.df_learningcurve = pd.read_csv(lc_path)
        else:
            self.compute_learning_curve(model = model, dv_path = dv_path, metric = metric, save_LC = save_LC, save_DV = save_DV)
        
        overall_performance = [self.evaluator_name]    

        ##### AREA UNDER THE LEARNING CURVE #####
        #Max - No contamination
        overall_performance.append(np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_max'].values, dx = 1/len(self.df_learningcurve)))
        #Max - True contamination
        overall_performance.append(np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_max'].values[:len(np.where(y_fakes_good==1)[0])],
                                            dx = 1/len(np.where(y_fakes_good==1)[0])))
        overall_performance.append(self.df_learningcurve[self.evaluator_name+'_lc_max'].values[len(np.where(y_fakes_good==1)[0])])
        #Min - No contamination
        overall_performance.append(-np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_min'].values, dx = 1/len(self.df_learningcurve)))
        #Min - True contamination
        overall_performance.append(-np.trapz(self.df_learningcurve[self.evaluator_name+'_lc_min'].values[:len(np.where(y_fakes_good==0)[0])],
                                             dx = 1/len(np.where(y_fakes_good==0)[0])))
        overall_performance.append(-self.df_learningcurve[self.evaluator_name+'_lc_min'].values[len(np.where(y_fakes_good==0)[0])])

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
        
        overall_performance.append(roc_auc_score(y_true=y_fakes_good, y_score = self.data_values_))
        
        overall_performance.append(self.cpu_time)
        overall_performance.append(self.random_state)
        self.df_overall_performance = pd.DataFrame(data = [overall_performance], columns = ['evaluator', 
                                                                                            'aulc_max','aulc_max_TrueCont', 'aulc_max_top33%',
                                                                                            'aulc_min', 'aulc_min_TrueCont', 'aulc_min_top66%',
                                                                                            'Acc_FakeRecog', 'Acc_FakeRecog_TrueCont',
                                                                                            'split_fakes', 'cpu_time', 'seed'])
        if save_OP:
            self.save_results_eval(self.df_overall_performance, path = self.path_save_results, name=model+"_"+metric+"_Evaluation")
        return self

    def compute_learning_curve(self, model: str = 'svm', metric: str = 'accuracy', dv_path: str = '', save_LC: bool = True, save_DV: bool = True) -> pd.DataFrame:
        
        randomidxs = np.arange(0, len(self.x_fakes))
        np.random.shuffle(randomidxs)
        self.x_fakes_sorted = self.x_fakes[randomidxs, :].cpu().numpy()
        if len(dv_path)>0:
            self.data_values_ = pd.read_csv(dv_path)[self.evaluator_name].values
        else:
            self.data_values_ = self.evaluate_data_values()
        if save_DV:
            self.save_results_dv(self.data_values_, path = self.path_save_results, 
                                 name='seed_'+str(self.random_state)+"_DataValues")
        print("Computing the learning curve following the metric's ordering")
        ranks = pd.DataFrame(self.data_values_[randomidxs]).rank(method = 'first', ascending = False).values
        self.df_learningcurve[self.evaluator_name+'_lc_max'] = self.evaluate_model_follow_ranking(ranks = ranks, model = model, metric = metric)

        print("Computing the learning curve following the metric's INVERSE ordering")
        ranks = pd.DataFrame(self.data_values_[randomidxs]).rank(method = 'first', ascending = True).values
        self.df_learningcurve[self.evaluator_name+'_lc_min'] = self.evaluate_model_follow_ranking(ranks = ranks, model = model, metric = metric)
        
        if save_LC:
            self.save_results_lc(df = self.df_learningcurve, path = self.path_save_results,
                                 name = model+"_"+metric+"_"+str(self.random_state)+"_LearningCurve")
        return

    def evaluate_model_follow_ranking(self, ranks: np.array, model: str = 'svm', metric: str = 'accuracy'):
        num_fakes = len(self.x_fakes_sorted)
        performance_metric = np.zeros(num_fakes+1, float)
        fake_index_added = []
        if model == 'svm':
            clf_orig = SVC(random_state=331, probability=True)
        elif model == 'rf':
            clf_orig = RandomForestClassifier(random_state=331)
        clf = clone(clf_orig).fit(self.x_train.cpu().numpy(),self.y_train.cpu().numpy())
        
        y_pred = clf.predict(self.x_test.cpu().numpy())
        yscores = clf.predict_proba(self.x_test.cpu().numpy())[:,1]
        if metric == 'f1_score':
            performance_metric[0] = f1_score(y_true=self.y_test.cpu().numpy(), y_pred=y_pred)
        elif metric == 'accuracy':
            performance_metric[0] = accuracy_score(y_true=self.y_test.cpu().numpy(), y_pred=y_pred)
        elif metric == 'auc':
            performance_metric[0] = roc_auc_score(y_true=self.y_test.cpu().numpy(), y_score=yscores)
            
        if self.verbose:
            rangelist = tqdm(np.arange(1, num_fakes+1))
        else:
            rangelist = np.arange(1, num_fakes+1)
            
        for rank_position in rangelist:
            idx_data_point = np.where(ranks==rank_position)[0]
            fake_index_added.append(idx_data_point)
            X_extended = np.concatenate((self.x_train.cpu().numpy(), self.x_fakes_sorted[np.sort(fake_index_added),:].reshape(len(fake_index_added),-1)))
            y_extended = np.concatenate((self.y_train.cpu().numpy(), np.ones(len(fake_index_added), dtype=int))).reshape(-1)
            clf = clone(clf_orig).fit(X_extended,y_extended)
            y_pred = clf.predict(self.x_test.cpu().numpy())
            yscores = clf.predict_proba(self.x_test.cpu().numpy())[:,1]
            if metric == 'f1_score':
                performance_metric[rank_position] = f1_score(y_true=self.y_test.cpu().numpy(), y_pred=y_pred)
            elif metric == 'accuracy':
                performance_metric[rank_position] = accuracy_score(y_true=self.y_test.cpu().numpy(), y_pred=y_pred)
            elif metric == 'auc':
                performance_metric[rank_position] = roc_auc_score(y_true=self.y_test.cpu().numpy(), y_score=yscores)
        return performance_metric

    def save_results_eval(self, df: pd.DataFrame, path: str, name: str):
        #evaluation
        try:
            dfread = pd.read_csv(path+self.formatted_date+"_"+name+".csv")
            #dfread = dfread.append(df, ignore_index = True)
            #dfread.to_csv(path+self.formatted_date+"_"+name+".csv", mode='w', header = True, index = False)
            df.to_csv(path+self.formatted_date+"_"+name+".csv", mode='a', header = False, index = False)
            print("Saved the evaluation results on the existing", path+self.formatted_date+"_"+name+".csv")
        except:
            df.to_csv(path+self.formatted_date+"_"+name+".csv", mode='w', header = True, index = False)
            print("Saved the evaluation results on the new", path+self.formatted_date+"_"+name+".csv")
        return self
    
    def save_results_lc(self, df: pd.DataFrame, path: str, name: str):
        #learning curve
        try:
            dfread = pd.read_csv(path+self.formatted_date+"_"+name+".csv")
            for col in df.columns:
                dfread[col] = df[col].values
            dfread.to_csv(path+self.formatted_date+"_"+name+".csv", mode='w', header = True, index = False)
            print("Saved the learning curve on the existing", path+self.formatted_date+"_"+name+".csv")
        except:
            df.to_csv(path+self.formatted_date+"_"+name+".csv", mode='w', header = True, index = False)
            print("Saved the learning curve on the new", path+self.formatted_date+"_"+name+".csv")
        return self

    def save_results_dv(self, data_values: np.array, path: str, name: str):
        #datavalues
        try:
            dfread = pd.read_csv(path+self.formatted_date+"_"+name+".csv")
            dfread[self.evaluator_name] = data_values.reshape(-1)
            dfread.to_csv(path+self.formatted_date+"_"+name+".csv", mode='w', header = True, index = False)
            print("Saved the datavalues on the existing", path+self.formatted_date+"_"+name+".csv")
        except:
            df = pd.DataFrame(data = data_values.reshape(-1), columns = [self.evaluator_name])
            df.to_csv(path+self.formatted_date+"_"+name+".csv", mode='w', header = True, index = False)
            print("Saved the datavalues on the new", path+self.formatted_date+"_"+name+".csv")
        return self


class ModelMixin:
    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor, metric:str):
        """Evaluate performance of the specified metric between label and predictions.

        Moves input tensors to cpu because of certain bugs/errors that arise when the
        tensors are not on the same device

        Parameters
        ----------
        y : torch.Tensor
            Labels to be evaluate performance of predictions
        y_hat : torch.Tensor
            Predictions of labels

        Returns
        -------
        float
            Performance metric
        """
        if len(metric)>0:
            self.metric = metric
            
        y = np.array([0 if x in [0,-1] else +1 for x in y.cpu().reshape(-1).numpy()])
        y_hat = y_hat.cpu().reshape(-1).numpy()
        if self.metric == 'auc':
            return roc_auc_score(y_true = y, y_score = y_hat) 
        elif self.metric == 'f1':
            ypred = np.array([0 if x <=0.5 else +1 for x in y_hat])
            return f1_score(y_true = y, y_pred = ypred, zero_division = 0.0) 
        elif self.metric == 'brier':
            return 1- brier_score_loss(y_true = y, y_prob=y_hat)
        else:
            print("Metric not in [auc, f1, brier]")
            return

    def input_model(self, pred_model: Model):
        """Input the prediction model.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        """
        self.pred_model = pred_model.clone()
        return self

    def input_metric(self, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        """Input the evaluation metric.

        Parameters
        ----------
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance
        """
        self.metric = metric
        return self

    def input_model_metric(
        self, pred_model: Model, metric: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        """Input the prediction model and the evaluation metric.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        return self.input_model(pred_model).input_metric(metric)



class ModelLessMixin:
    """Mixin for DataEvaluators without a prediction model and use embeddings.

    Using embeddings and then predictiong the data values has been used by
    Ruoxi Jia Group with their KNN Shapley and LAVA data evaluators.

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Attributes
    ----------
    embedding_model : Model
        Embedding model used by model-less DataEvaluator to compute the data values for
        the embeddings and not the raw input.
    pred_model : Model
        The pred_model is unused for training, but to compare a series of models on
        the same algorithim, we compare against a shared prediction algorithim.
    """

    def embeddings(
        self, *tensors: tuple[Union[Dataset, torch.Tensor], ...]
    ) -> tuple[torch.Tensor, ...]:
        """Returns Embeddings for the input tensors

        Returns
        -------
        tuple[torch.Tensor, ...]
            Returns tupple of tensors equal to the number of tensors input
        """
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            return tuple(self.embedding_model.predict(tensor) for tensor in tensors)

        # No embedding is used
        return tensors
    
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