# I have modified the existing file
import json
import warnings
from itertools import accumulate, chain
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset, Subset
from collections import Counter
from opendataval.dataloader.register import Register
from opendataval.dataloader.util import CatDataset
from sklearn.cluster import KMeans
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8##

import os
from PIL import Image
from tqdm import tqdm
#import clip


Self = TypeVar("Self", bound="DataFetcher")

class DataFetcher:
    def __init__(
        self,
        dataset_name: str,
        random_state: Optional[int] = 331,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        verbose: Optional[bool] = True,
    ):
        self.dataset = dataset_name
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.check_path()
        
    def setup(self, 
              train_anom_prop: float = 0.02,
              fake_anom_prop: float = 0.24,
              test_anom_prop: float = 0.5,
              ):
        
        if self.verbose:
            print("Downloading CLIP ...")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.clip_model, self.clip_process = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
        
        if self.verbose:
            print("Extracting Features using CLIP ...")
            
        X_ok, y_ok = self.create_image_dataset(image_folder = '/OK', label_value = -1)
        X_nok, y_nok = self.create_image_dataset(image_folder = '/NOK', label_value = 1)
        self.X = torch.cat([X_ok, X_nok])
        self.y = torch.cat([y_ok, y_nok]).reshape(-1)

        self.num_anomalies = len(y_nok)
        self.num_normals = len(y_ok)
        self.num_points = self.X.shape[0]
        self.covar_dim = self.X.shape[1]
        
        if self.verbose:
            print("Labels of images:", Counter(self.y.cpu().numpy()), "Shape of (encoded) images:", self.X.shape)
        
        return self.split_dataset_by_prop(train_prop = train_anom_prop, fake_prop = fake_anom_prop, test_prop = test_anom_prop)

    #@classmethod
    def from_data_splits(
        self,
        x_train: Union[Dataset, np.ndarray],
        y_train: np.ndarray,
        x_fake: Union[Dataset, np.ndarray],
        y_fake: np.ndarray,
        x_test: Union[Dataset, np.ndarray],
        y_test: np.ndarray):
        """Return DataFetcher from already split data.

        Parameters
        ----------
        x_train : Union[Dataset, np.ndarray]
            Input training covariates
        y_train : np.ndarray
            Input training labels
        x_valid : Union[Dataset, np.ndarray]
            Input validation covariates
        y_valid : np.ndarray
            Input validation labels
        x_test : Union[Dataset, np.ndarray]
            Input testing covariates
        y_test : np.ndarray
            Input testing labels
        one_hot : bool
            Whether the label data has already been one hot encoded. This is just a flag
            and not transform will be applied
        random_state : RandomState, optional
            Initial random state, by default None

        Raises
        ------
        ValueError
            Loaded Data set covariates and labels must be of same length.
        ValueError
            All covariates must be of same dimension.
            All labels must be of same dimension.
        """
        self._presplit_data(x_train, x_fake, x_test, y_train, y_fake, y_test)

        return self

    def _presplit_data(self, x_train, x_fake, x_test, y_train, y_fake, y_test):
        if not len(x_train) == len(y_train):
            raise ValueError("Training Covariates and Labels must be of same length.")
        if not len(x_fake) == len(y_fake):
            raise ValueError("Fake Covariates and Labels must be of same length.")
        if not len(x_test) == len(y_test):
            raise ValueError("Testing Covariates and Labels must be of same length.")

        if not (x_train[0].shape ==  x_fake[0].shape == x_test[0].shape):
            raise ValueError("Covariates must be of same shape.")
        if not (y_train[0].shape == y_fake[0].shape == y_test[0].shape):
            raise ValueError("Labels must be of same shape.")

        self.x_train, self.x_fake, self.x_test = x_train, x_fake, x_test
        self.y_train, self.y_fake, self.y_test = y_train, y_fake, y_test
        self.contamination = len(torch.where(y_train==1)[0])/len(y_train)
        
        tr, fake, test = len(self.x_train), len(self.x_fake), len(self.x_test)
        self.train_indices = np.fromiter(range(tr), dtype=int)
        self.fake_indices = np.fromiter(range(tr , tr +fake), dtype=int)
        self.test_indices = np.fromiter(range(tr +  fake, tr  + fake + test), dtype=int)
        
        if self.verbose:
            print("------------------------------------------------------------------------------------------------------")
            print("Summary of the dataset's characteristics for", self.dataset)
            print("Sizes after splitting are the following:")
            print("- Training:", self.x_train.shape, "with # anomalies:", Counter(self.y_train)[1])
            #print("- Validation:", self.x_valid.shape, "with # anomalies:", Counter(self.y_valid)[1])
            print("- Fakes:", self.x_fake.shape, "with unknown category of anomalies")
            print("- Test:", self.x_test.shape, "with # anomalies:", Counter(self.y_test)[1])
            print("------------------------------------------------------------------------------------------------------")

        return
    
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
        return self.x_train, self.y_train, self.x_fake, self.y_fake, self.x_test, self.y_test

    def split_dataset_by_prop(
        self,
        train_prop: float = 0.0,
        fake_prop: float = 0.0,
        test_prop: float = 0.0,
    ):
        """Split the covariates and labels to the specified proportions."""
        train_count, fake_count, test_count = (round(self.num_anomalies * p) for p in (train_prop, fake_prop, test_prop))
        
        #Constraint on the count of anomalies
        
        fake_count = min(fake_count, 200)
        train_count = min(train_count, 20)
        test_count = min(test_count, 500)
        
        return self.split_dataset_by_count(train_count, fake_count, test_count)

    def split_dataset_by_count(
        self,
        train_count: int = 0,
        #valid_count: int = 0,
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
        print("Expected counts:", train_count, fake_count, test_count)
        
        if sum((train_count, fake_count, test_count)) > self.num_anomalies:
            raise ValueError(f"Split totals must be < {self.num_anomalies=} and of the same type: ")
        
        if self.verbose:
            print("Splitting the Dataset into Training / Test / Fakes...")

        remaining_anomalies = torch.nonzero(self.y == 1).squeeze()
        remaining_normals = torch.nonzero(self.y == -1).squeeze()
        
        #print("@@@@@ Step 1: ", len(remaining_anomalies), len(remaining_normals))
        # Creating the test set
        #try:
        test_idx_anomalies = torch.randperm(remaining_anomalies.size(0))[:test_count]
        test_idx_normals = torch.randperm(remaining_normals.size(0))[:test_count]
        test_idx = torch.cat([remaining_normals[test_idx_normals], remaining_anomalies[test_idx_anomalies]]).clone()
        remaining_anomalies = remaining_anomalies[~torch.isin(remaining_anomalies, test_idx)]
        remaining_normals = remaining_normals[~torch.isin(remaining_normals, test_idx)]

        clustered_anom = KMeans(n_clusters=10, random_state=self.random_state, n_init='auto').fit(self.X[remaining_anomalies, :].cpu()).labels_
        sort_by_cluster = np.argsort(clustered_anom)
        train_anomalies_idx = remaining_anomalies[sort_by_cluster[:train_count]].clone()
        remaining_anomalies = remaining_anomalies[~torch.isin(remaining_anomalies, train_anomalies_idx)]

        # Good Fakes
        fake_anomalies_idx = remaining_anomalies[torch.randperm(remaining_anomalies.size(0))[:fake_count]].clone()
        remaining_anomalies = remaining_anomalies[~torch.isin(remaining_anomalies, fake_anomalies_idx)]
        
        if len(remaining_anomalies)> len(fake_anomalies_idx):
            noise_anomalies_idx = remaining_anomalies[torch.randperm(remaining_anomalies.size(0))[:len(fake_anomalies_idx)]].clone()
        else:
            noise_anomalies_idx = remaining_anomalies.clone()

        try:
            polluted_anomalies_idx = remaining_normals[torch.randperm(remaining_normals.size(0))[:len(fake_anomalies_idx)]].clone()
        except:
            print("Not enough normals to use as polluted!")
            return
        remaining_normals = remaining_normals[~torch.isin(remaining_normals, polluted_anomalies_idx)]

        # Training Normal Examples
        if len(remaining_normals)>500:
            training_normals_idx = remaining_normals[torch.randperm(remaining_normals.size(0))[:500]].clone()
        else:
            training_normals_idx = remaining_normals.clone() #

        # Validation Normal Examples
        #remaining_normals = remaining_normals[~torch.isin(remaining_normals, training_normals_idx)]
        #valid_normals_idx = remaining_normals.clone()

        # Compose X, y for each set of examples

        # Training set
        self.x_train = torch.cat([self.X[training_normals_idx, :], self.X[train_anomalies_idx, :]])
        self.y_train = torch.cat([self.y[training_normals_idx], self.y[train_anomalies_idx]])
        self.contamination = len(train_anomalies_idx)/len(training_normals_idx)
        
        # Validation set
        #self.x_valid = torch.cat([self.X[valid_normals_idx, :], self.X[valid_anomalies_idx, :]])
        #self.y_valid = torch.cat([self.y[valid_normals_idx], self.y[valid_anomalies_idx]])

        # Fake set
        if self.verbose:
            print("Transforming the selected fakes into Noise using strong augmentation ...")
        transformation_list = [T.ElasticTransform(alpha=2000.0)]
        #print(noise_anomalies_idx, len(noise_anomalies_idx))
        X_noise_fake, y_noise_fake = self.create_noise_fakes('/NOK', noise_anomalies_idx, transformation_list)
        self.x_fake = torch.cat([self.X[fake_anomalies_idx, :], self.X[polluted_anomalies_idx, :], X_noise_fake])
        self.y_fake = torch.ones(len(self.x_fake), dtype=torch.int, device=self.device)
        self.y_fake_true_label = torch.cat([self.y[fake_anomalies_idx], self.y[polluted_anomalies_idx], y_noise_fake])

        # Testing set
        self.x_test = self.X[test_idx, :]
        self.y_test = self.y[test_idx]

        if self.verbose:
            print("------------------------------------------------------------------------------------------------------")
            print("Summary of the dataset's characteristics for", self.dataset)
            print("Initial size:", self.X.shape, "with #anomalies:", len(torch.where(self.y==1)[0]))
            print("\n")
            print("Sizes after splitting are the following:")
            print("- Training:", self.x_train.shape, "with # anomalies:", Counter(self.y_train.cpu().numpy())[1])
            #print("- Validation:", self.x_valid.shape, "with # anomalies:", Counter(self.y_valid.cpu().numpy())[1])
            print("- Fakes:", self.x_fake.shape, "with # correct anomalies:", len(fake_anomalies_idx), 
                  "("+str(100*len(fake_anomalies_idx)/len(self.y_fake))[:4]+"%) # polluted anomalies:",
                  len(polluted_anomalies_idx),"("+str(100*len(polluted_anomalies_idx)/len(self.y_fake.cpu().numpy()))[:4]+"%) and # noise anomalies:", 
                  len(y_noise_fake), "("+str(100*len(y_noise_fake)/len(self.y_fake))[:4]+"%)")
            print("- Test:", self.x_test.shape, "with # anomalies:", Counter(self.y_test.cpu().numpy())[1])
            print("------------------------------------------------------------------------------------------------------")
        return self

    def export_dataset(
        self,
        covariates_names: list[str],
        labels_names: list[str],
        output_directory: Path = Path.cwd(),
    ):
        if isinstance(covariates_names, str):
            covariates_names = [covariates_names]
        if isinstance(labels_names, str):
            labels_names = [labels_names]
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            output_directory.mkdir(parents=True)

        columns = covariates_names + labels_names
        x_train, x_fake, x_test = self.x_train, self.x_fake, self.x_test
        y_train, y_fake, y_test = self.y_train, self.y_fake, self.y_test

        if self.one_hot:
            y_train = np.argmax(y_train, axis=1, keepdims=True) if y_train.size else []
            y_fake = np.argmax(y_fake, axis=1, keepdims=True) if y_fake.size else []
            y_test = np.argmax(y_test, axis=1, keepdims=True) if y_test.size else []

        def generate_data(covariates, labels):
            data = CatDataset(covariates, labels)
            for cov, lab in DataLoader(data, batch_size=1, shuffle=False):
                yield from np.hstack((cov, lab))

        def save_to_csv(data, file_name):
            file_path = output_directory / file_name
            pd.DataFrame(data, columns=columns).to_csv(file_path, index=False)

        save_to_csv(generate_data(x_train, y_train), "train.csv")
        save_to_csv(generate_data(x_fake, x_fake), "fake.csv")
        save_to_csv(generate_data(x_test, y_test), "test.csv")

        noisy_indices = (
            self.noisy_train_indices.tolist()
            if hasattr(self, "noisy_train_indices")
            else []
        )
        out_path = output_directory / f"noisy-indices-{self.dataset.dataset_name}.json"
        with open(out_path, "w+") as f:
            json.dump(noisy_indices, f)

        return    

    def create_image_dataset(self, image_folder: str, label_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        labels = []
        image_files = [f for f in os.listdir(self.path+image_folder) if f.lower().endswith(('.png'))]

        for image_file in image_files:
            image_path = os.path.join(self.path+image_folder, image_file)

            try:
                image = Image.open(image_path).convert("RGB")
                processed_image = self.clip_process(image)
                images.append(processed_image)
                labels.append([label_value])

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

        y = torch.tensor(labels, dtype=torch.int, device=self.device).view(-1)
        X = self.encode_images_nolabels(DataLoader(images, batch_size=50, shuffle=False))

        return X, y


    def encode_images_nolabels(self, dataloader):
        img_features = []
        with torch.no_grad():
            for img in dataloader:
                features = self.clip_model.encode_image(img.to(self.device))
                img_features.append(features)
        return torch.cat(img_features)

    def create_noise_fakes(self, image_folder: str, noise_anomalies_idx, transformation_list):
        image_files = [f for f in os.listdir(self.path+image_folder) if f.lower().endswith(('.png'))]
        processed_imgs = []
        for j,image_file in enumerate(image_files):
            image_path = os.path.join(self.path+image_folder, image_file)
            # Open the image using PIL
            if j +self.num_normals in noise_anomalies_idx:
                image = Image.open(image_path).convert('RGB')
                for transform in transformation_list:
                    tfm = self._transform_aug(224,transform)
                    img = tfm(image)
                    processed_imgs.append(img)
        X_noise_fake = self.encode_images_nolabels(DataLoader(processed_imgs, batch_size=50, shuffle=False))
        y_noise_fake = 2*torch.ones(len(noise_anomalies_idx), dtype=torch.int, device=self.device)
        return X_noise_fake, y_noise_fake
        
    def _transform_aug(self, n_px, transform):
        return T.Compose([T.Resize(n_px, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(n_px), transform, lambda image: image.convert("RGB"),
                          T.ToTensor(), T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])   
        
    def check_path(self):
        
        if self.dataset == 'PEG':
            self.path = '/home/pel2rng/Desktop/python/data_files/PEG_Folder1234'
        elif self.dataset == 'R0002':
            self.path = '/home/pel2rng/Desktop/python/data_files/R0002_20150528_D20_P00'
        elif self.dataset == 'R0033':
            self.path = '/home/pel2rng/Desktop/python/data_files/R0033_20151023_RollenschuhFlaeche'
        else:
            print("The dataset is not registered!")
        return self