from typing import Dict, List, Optional

from sklearn.base import BaseEstimator

from qolmat.imputations.imputers import ImputerRegressor, ImputerGenerativeModel
from qolmat.utils.exceptions import PytorchNotInstalled

# try:
#     from tensorflow.keras.callbacks import EarlyStopping
# except ModuleNotFoundError:
#     raise PytorchNotInstalled


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
        **hyperparams,
    ):
        super().__init__(
            groups=groups, model=model, **hyperparams
        )
        self.epochs = epochs
        self.batch_size = batch_size

    def get_params_fit(self) -> Dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size}
