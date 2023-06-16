from typing import Dict, List, Optional, Tuple

import math

from sklearn.base import BaseEstimator
from sklearn import preprocessing

import numpy as np
import pandas as pd
import time
import gc

import torch
from torch.utils.data import DataLoader, TensorDataset


from qolmat.imputations.imputers import ImputerRegressor, ImputerGenerativeModel
from qolmat.utils.exceptions import PytorchNotInstalled
from qolmat.benchmark import missing_patterns, metrics

# try:
#     from tensorflow.keras.callbacks import EarlyStopping
# except ModuleNotFoundError:
#     raise PytorchNotInstalled


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params


class ImputerRegressorPytorch(ImputerRegressor):
    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "column",
        epochs: int = 100,
        batch_size: int = 100,
        **hyperparams,
    ):
        super().__init__(
            groups=groups, estimator=estimator, handler_nan=handler_nan, **hyperparams
        )
        self.epochs = epochs
        self.batch_size = batch_size

    def get_params_fit(self) -> Dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size}


class ImputerGenerativeModelPytorch(ImputerGenerativeModel):
    def __init__(
        self,
        groups: List[str] = [],
        model: Optional[BaseEstimator] = None,
        epochs: int = 100,
        batch_size: int = 100,
        print_valid: bool = False,
        **hyperparams,
    ):
        super().__init__(groups=groups, model=model, **hyperparams)
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_valid = print_valid

    def get_params_fit(self) -> Dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "print_valid": self.print_valid,
        }
