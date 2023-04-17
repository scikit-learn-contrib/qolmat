from typing import Dict, List, Optional
from qolmat.imputations.imputers import ImputerRegressor
from sklearn.base import BaseEstimator
from tensorflow.keras.callbacks import EarlyStopping


class ImputerRegressorKeras(ImputerRegressor):
    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "column",
        epochs: int = 100,
        monitor: str = "loss",
        patience: int = 5,
        **hyperparams,
    ):
        super().__init__(
            groups=groups, estimator=estimator, handler_nan=handler_nan, **hyperparams
        )
        self.epochs = epochs
        self.monitor = monitor
        self.patience = patience

    def get_params_fit(self) -> Dict:
        es = EarlyStopping(monitor=self.monitor, patience=self.patience, verbose=False, mode="min")
        return {"epochs": self.epochs, "callbacks": [es]}
