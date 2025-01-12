# I have modified the existing file -- almost everything is different
import json
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter
from opendataval.dataloader.util import CatDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
import os
import warnings


Self = TypeVar("Self", bound="DataFetcherTD")
warnings.filterwarnings("ignore")

class DataFetcherTD:
    def __init__(
        self,
        X: np.array,
        y: np.array,
        datasets_path: str = '',
        n_auxiliary_ds: int = 5,
        max_size: int = 1000,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        verbose: Optional[bool] = True,
    ):
        self.X = torch.from_numpy(X).to(device, dtype=torch.float32)
        self.y = torch.from_numpy(y).to(device)
        self.datasets_path = datasets_path
        self.device = device
        self.verbose = verbose
        self.random_state = random_state      
        self.n_auxiliary_ds = n_auxiliary_ds
        self.max_size = max_size
        
    def setup(self, 
              train_anom_prop: float = 0.1,
              fake_anom_prop: float = 0.4,
              test_anom_prop: float = 0.5,
              ):
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)            
            
        self.y[self.y==0] = -1
        
        self.setup_auxiliary_datasets()
        
        self.num_anomalies = len(torch.where(self.y == 1)[0])
        self.num_normals = len(torch.where(self.y != 1)[0])
        self.num_points = self.X.shape[0]
        self.covar_dim = self.X.shape[1]
        
        return self.split_dataset_by_prop(train_prop = train_anom_prop, fake_prop = fake_anom_prop, test_prop = test_anom_prop)

    #@property
    def datapoints(self):
        """Return split data points to be input into a DataEvaluator as tensors.

        Returns
        -------
        (torch.Tensor | Dataset, torch.Tensor)
            Training Covariates, Training Labels
        (torch.Tensor | Dataset, torch.Tensor)
            Validation Covariates, Valid Labels
        (torch.Tensor | Dataset, torch.Tensor)
            Test Covariates, Test Labels
        """
        return self.x_train, self.y_train, self.x_fake, self.y_fake, self.y_fake_true_label, self.x_test, self.y_test

    def split_dataset_by_prop(
        self,
        train_prop: float = 0.0,
        fake_prop: float = 0.0,
        test_prop: float = 0.0,
    ):
        """Split the covariates and labels to the specified proportions."""
        train_count, fake_count, test_count = (round(self.num_anomalies * p) for p in (train_prop, fake_prop, test_prop))

        test_count = int(max(min(test_count, 250), 50)) #no more than 250 and at least 75
        train_count = int(max(min(train_count, 50), 5)) #no more than 50 and at least 5
        fake_count = int(min(fake_count, 250, self.num_anomalies-test_count-train_count))
        
        
        return self.split_dataset_by_count(train_count, fake_count, test_count)

    def split_dataset_by_count(
        self,
        train_count: int = 0,
        fake_count: int = 0,
        test_count: int = 0,
    ):
        """Split the covariates and labels to the specified counts.

        Parameters
        ----------
        train_count : int
            Number/proportion training points
        valid_count : int
            Number/proportion validation points
        test_count : int
            Number/proportion test points

        Returns
        -------
        self : object
            Returns a DataFetcher with covariates, labels split into train/valid/test.

        Raises
        ------
        ValueError
            Invalid input for splitting the data set, either the proportion is more
            than 1 or the total splits are greater than the len(dataset)
        """        
        if sum((train_count, fake_count, test_count)) > self.num_anomalies:
            raise ValueError(f"Split totals must be < {self.num_anomalies=} and of the same type: ")
        
        if self.verbose:
            print("Splitting the Dataset into Training / Test / Fakes...")

        remaining_anomalies = torch.nonzero(self.y == 1).squeeze()
        remaining_normals = torch.nonzero(self.y == -1).squeeze()

        test_idx_anomalies = torch.randperm(remaining_anomalies.size(0))[:test_count]
        test_idx_normals = torch.randperm(remaining_normals.size(0))[:test_count]
        test_idx = torch.cat([remaining_normals[test_idx_normals], remaining_anomalies[test_idx_anomalies]]).clone()
        
        remaining_anomalies = remaining_anomalies[~torch.isin(remaining_anomalies, test_idx)]
        remaining_normals = remaining_normals[~torch.isin(remaining_normals, test_idx)]

        clustered_anom = KMeans(n_clusters=min(10,len(remaining_anomalies)), random_state=self.random_state, n_init='auto').fit(self.X[remaining_anomalies, :].cpu()).labels_
        sort_by_cluster = np.argsort(clustered_anom)
        train_anomalies_idx = remaining_anomalies[sort_by_cluster[:train_count]].clone()
        remaining_anomalies = remaining_anomalies[~torch.isin(remaining_anomalies, train_anomalies_idx)]

        # Good Fakes
        fake_anomalies_idx = remaining_anomalies[torch.randperm(remaining_anomalies.size(0))[:fake_count]].clone()
        
        # Polluted Fakes
        polluted_anomalies_idx = remaining_normals[torch.randperm(remaining_normals.size(0))[:fake_count]].clone()
        if len(polluted_anomalies_idx) != fake_count:
            print("Not enough normals to use as polluted! Expected", fake_count,"but used only", len(polluted_anomalies_idx))
            
        remaining_normals = remaining_normals[~torch.isin(remaining_normals, polluted_anomalies_idx)]

        # Training Normal Examples
        if len(remaining_normals)>1000:
            training_normals_idx = remaining_normals[torch.randperm(remaining_normals.size(0))[:1000]].clone()
        else:
            training_normals_idx = remaining_normals.clone() #

        # Training set
        self.x_train = torch.cat([self.X[training_normals_idx, :], self.X[train_anomalies_idx, :]]).to(dtype=torch.float32)
        self.y_train = torch.cat([self.y[training_normals_idx], self.y[train_anomalies_idx]])
        self.contamination = len(train_anomalies_idx)/self.y_train.size(0)
        
        means = torch.mean(self.x_train, dim=0, keepdim=True).clone()
        stds = torch.std(self.x_train, dim=0, keepdim=True).clone()
        stds[stds<0.001] = 1
        self.x_train = (self.x_train - means) / stds

        # Fake set
        _, self.X_aux, _, self.y_aux = train_test_split(self.X_aux, self.y_aux, test_size=fake_count, random_state=self.random_state, stratify = self.y_aux)
        self.X_aux, self.y_aux = torch.from_numpy(self.X_aux).to(self.device, dtype=torch.float32), torch.from_numpy(self.y_aux).to(self.device)
        
        self.x_fake = torch.cat([self.X[fake_anomalies_idx, :], self.X[polluted_anomalies_idx, :], self.X_aux]).to(dtype=torch.float32)
        self.y_fake = torch.ones(len(self.x_fake), dtype=torch.int, device=self.device)
        self.y_fake_true_label = torch.cat([self.y[fake_anomalies_idx], self.y[polluted_anomalies_idx], 
                                            2*torch.ones(len(self.y_aux), dtype=torch.int, device=self.device)])
        
        self.x_fake = (self.x_fake - means) / stds

        # Testing set
        self.x_test = self.X[test_idx, :].to(dtype=torch.float32)
        self.y_test = self.y[test_idx]
        self.x_test = (self.x_test - means) / stds
        
        if self.verbose:
            print("\n")
            print("------------------------------------------------------------------------------------------------------")
            print("Summary of the dataset's characteristics")
            print("Initial shape:", self.X.cpu().numpy().shape, "with #anomalies:", len(torch.where(self.y==1)[0]))
            print("\n")
            print("Shapes after splitting are the following:")
            print("- Training:", self.x_train.cpu().numpy().shape, "with # anomalies:", Counter(self.y_train.cpu().numpy())[1])
            print("- Fakes:", self.x_fake.cpu().numpy().shape, "with # correct anomalies:", len(fake_anomalies_idx), 
                  "("+str(100*len(fake_anomalies_idx)/len(self.y_fake))[:4]+"%) # polluted anomalies:",
                  len(polluted_anomalies_idx),"("+str(100*len(polluted_anomalies_idx)/len(self.y_fake.cpu().numpy()))[:4]+"%) and # noise anomalies:", 
                  len(self.y_aux), "("+str(100*len(self.y_aux)/len(self.y_fake))[:4]+"%)")
            print("- Test:", self.x_test.cpu().numpy().shape, "with # anomalies:", Counter(self.y_test.cpu().numpy())[1])
            print("------------------------------------------------------------------------------------------------------")
            print("\n")
            
        self.index_fake = np.concatenate((fake_anomalies_idx.cpu().numpy(), polluted_anomalies_idx.cpu().numpy()))
        self.index_test = test_idx.cpu().numpy()
        self.index_train = np.concatenate((training_normals_idx, train_anomalies_idx))
        return self
    
    
    def setup_auxiliary_datasets(self):
        X_aux = np.empty(shape=(0,self.X.shape[1]), dtype= float)
        y_aux = np.empty(shape=(0), dtype= int)
        self.index_aux = {}

        auxiliary_ds = np.random.choice([f for f in os.listdir(self.datasets_path) if f.endswith('.npz')], self.n_auxiliary_ds, replace = False)

        self.auxiliary_ds_paths = [self.datasets_path+x for x in auxiliary_ds]
        self.auxiliary_ds_names = [self.auxiliary_ds_paths[i].split('_')[-1].split('.npz')[0] for i in range(self.n_auxiliary_ds)]
        
        for j,ds_path in enumerate(self.auxiliary_ds_paths):
            
            data = np.load(ds_path, allow_pickle=True)
            X, y = data['X'], data['y']
            c = Counter(y)
            if c[1]>int(self.max_size//8):
                anom_idx = np.random.choice(np.where(y==1)[0], int(self.max_size//8), replace = False)
            else:
                anom_idx = np.where(y==1)[0]
            if c[0]>int(self.max_size//8):
                norm_idx = np.random.choice(np.where(y==0)[0], int(min(c[0], self.max_size//4 - len(anom_idx))), replace = False)
            else:
                norm_idx = np.where(y==0)[0]
            self.index_aux[self.auxiliary_ds_names[j]] = np.concatenate((norm_idx, anom_idx))
            y = np.concatenate((y[norm_idx], y[anom_idx]))
            X = np.concatenate((X[norm_idx, :], X[anom_idx, :]))
            y+=2*j
            X_aux = np.concatenate((X_aux, self.augment_features(X)))
            y_aux = np.concatenate((y_aux, y))
        self.X_aux = X_aux
        self.y_aux = y_aux
        
        return self
    
    def augment_features(self, X_aux: np.array):
        
        target_dimension = self.X.shape[1]
        if X_aux.shape[1] == target_dimension:
            return X_aux
        elif min(X_aux.shape) > target_dimension:
            return PCA(n_components=target_dimension, random_state= self.random_state).fit_transform(X_aux)
        elif 3*min(X_aux.shape) >= target_dimension:
            X1 = PCA(n_components=min(X_aux.shape), random_state= self.random_state).fit_transform(X_aux)
            X2 = FeatureAgglomeration(n_clusters=min(X_aux.shape)).fit_transform(X_aux)
            Xconcat = np.concatenate((X_aux, X1, X2), axis = 1)
            return Xconcat[:, :target_dimension]
        else:
            X1 = PCA(n_components=min(X_aux.shape), random_state= self.random_state).fit_transform(X_aux)
            X2 = FeatureAgglomeration(n_clusters=min(X_aux.shape)).fit_transform(X_aux)
            X3 = GaussianRandomProjection(n_components=target_dimension-X1.shape[1]-X2.shape[1],random_state=self.random_state).fit_transform(X_aux)
            Xconcat = np.concatenate((X1, X2, X3), axis = 1)
            return Xconcat[:, :target_dimension]
                
        