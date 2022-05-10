from audioop import rms
from lib2to3.pytree import Base
from math import floor
from signal import signal
from typing import Optional, Union
import datetime
import numpy as np
import pandas as pd
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state, resample


def eval(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared = False)
        mae = mean_absolute_error(y_true, y_pred)
        
        y_true_nan = np.where(y_true > 0, y_true, np.nan)
        y_pred_nan = np.where(y_true > 0, y_pred, np.nan)

        mape = np.nanmean(np.abs((y_true_nan - y_pred_nan)/np.abs(y_true_nan)))
        wmape = np.mean(np.abs(y_true - y_pred))/np.mean(np.abs(y_true))
        return rmse, mae, mape, wmape


class EvaluateImputor:

    def __init__(self,
                signal: NDArray,
                indices_to_nan,
                prop: float = 0.05,
                cv: int = 3,
                random_state: Optional[Union[int, RandomState]] = None
                ):
                
                self.signal = signal
                random_state = check_random_state(random_state)
                
                nan_subsets = []
                
                for _ in range(cv):
                    nan_subsets.append(
                            resample(
                                indices_to_nan,
                                replace=False,
                                n_samples=floor(len(indices_to_nan) * prop),
                                random_state=random_state,
                                stratify=None,
                        )
                    )
                self.nan_subsets = nan_subsets
                self.random_state = random_state

    def scores_imputor(self, imputor: BaseEstimator, func = eval):

        input = self.signal.values.copy()
        transform = self.signal.values.copy()
        
        RMSE=[]
        MAE=[]
        MAPE=[]
        WMAPE=[]

        for nan_indices in self.nan_subsets:
            transform.flat[nan_indices] = np.nan
            impute, _, _ = imputor.fit_transform(signal =transform)
            rmse, mae, mape, wmape = func(input, impute)
            RMSE.append(rmse)
            MAE.append(mae)
            MAPE.append(mape)
            WMAPE.append(wmape)
        print(f"call={datetime.datetime.now().strftime('%H:%M')}")
        return {"rmse" : np.mean(RMSE),
                "mae" : np.mean(MAE),
                "mape" : np.mean(MAPE),
                "wmape" : np.mean(WMAPE)}