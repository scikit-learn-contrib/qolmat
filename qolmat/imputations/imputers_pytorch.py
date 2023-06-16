from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from qolmat.imputations.imputers import ImputerRegressor, ImputerGenerativeModel
from qolmat.utils.exceptions import PytorchNotInstalled
from qolmat.benchmark import metrics

try:
    import torch
except ModuleNotFoundError:
    raise PytorchNotInstalled


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
        x_valid: pd.DataFrame = None,
        x_valid_mask: pd.DataFrame = None,
        print_valid: bool = False,
        metrics_valid: Dict[str, Callable] = {"mae": metrics.mean_absolute_error},
        **hyperparams,
    ):
        super().__init__(groups=groups, model=model, **hyperparams)
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_valid = x_valid
        self.x_valid_mask = x_valid_mask
        self.print_valid = print_valid
        self.metrics_valid = metrics_valid

    def get_params_fit(self) -> Dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "x_valid": self.x_valid,
            "x_valid_mask": self.x_valid_mask,
            "print_valid": self.print_valid,
            "metrics_valid": self.metrics_valid,
        }
