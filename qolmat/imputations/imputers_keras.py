from typing import Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator

from qolmat.imputations.imputers import Imputer, ImputerRegressor
from qolmat.utils.exceptions import KerasExtraNotInstalled

try:
    from tensorflow.keras.callbacks import EarlyStopping
except ModuleNotFoundError:
    raise KerasExtraNotInstalled


class ImputerRegressorKeras(ImputerRegressor):
    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "column",
        epochs: int = 100,
        monitor: str = "loss",
        patience: int = 5,
    ):
        Imputer.__init__(
            self,
            imputer_params=("handler_nan", "epochs", "monitor", "patience"),
            groups=groups,
        )
        self.epochs = epochs
        self.monitor = monitor
        self.patience = patience
        self.estimator = estimator
        self.handler_nan = handler_nan

    def get_params_fit(self) -> Dict:
        es = EarlyStopping(monitor=self.monitor, patience=self.patience, verbose=False, mode="min")
        return {"epochs": self.epochs, "callbacks": [es]}
