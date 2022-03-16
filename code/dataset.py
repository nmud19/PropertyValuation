import pandas as pd
import numpy as np
from scipy.stats import skew
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split,Dataset
from torch.utils.data import Dataset
from typing import Callable, List

class HousePricesDataSet(Dataset) : 
    """Dataset class to read the house price dataset class"""
    def __init__(
        self, 
        train_data_dir:str,
        feature_engg_func:Callable
        )->None : 
        """Init function to load the df, has hardcoded feature engineering for now"""
        super().__init__()
        df = pd.read_csv(train_data_dir)
        y_var = 'SalePrice'
        
        # Create xvars and y
        self.x_tr = feature_engg_func(df.drop(columns=y_var))
        print("NUM FEATS : ",self.x_tr.shape)
        self.y_tr = df[y_var]
        self.x_tr = torch.tensor(self.x_tr.values).float()
        self.y_tr = torch.tensor(self.y_tr.values)
    
    def __len__(self) -> int: 
        """Get the shape of train data"""
        return len(self.x_tr)
    
    def __getitem__(self, idx) -> List[torch.tensor]:
        """Return the row"""
        return [self.x_tr[idx], self.y_tr[idx]] 
    