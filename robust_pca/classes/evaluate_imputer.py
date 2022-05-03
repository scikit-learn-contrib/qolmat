from audioop import rms
from lib2to3.pytree import Base
from math import floor
from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state, resample

class evaluate_imputor:

    def __init__(self,
                signal: ArrayLike,
                prop: float = 0.1,
                cv = 3,
                random_state: Optional[Union[int, RandomState]] = None):

                self.signal = signal
                indices = signal.loc[~np.isnan(signal) & signal > 0].index
                random_state = check_random_state(random_state)
                nan_subsets = []
                for _ in range(cv):
                    nan_subsets.append(
                            resample(
                                indices,
                                replace=False,
                                n_samples=floor(len(indices) * prop),
                                random_state=random_state,
                                stratify=None,
                        )
                    )
                self.nan_subsets = nan_subsets
                self.random_state = random_state

    def eval(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared = False)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred)/np.abs(y_true)))
        wmape = np.mean(np.abs(y_true - y_pred))/np.mean(np.abs(y_true))
        return rmse, mae, mape, wmape

    def scores_imputor(self, imputor: BaseEstimator, func = eval):
        transform_signal = self.signal.copy()
        RMSE=[]
        MAE=[]
        MAPE=[]
        WMAPE=[]

        for nan_indices in self.nan_subsets:
            transform_signal.iloc[nan_indices] = np.nan
            transform_signal = imputor.fit_transform(transform_signal)
            rmse, mae, mape, wmape = func(self.signal, transform_signal)
            RMSE.append(rmse)
            MAE.append(mae)
            MAPE.append(mape)
            WMAPE.append(wmape)

            return {"rmse" : np.mean(RMSE),
                    "mae" : np.mean(MAE),
                    "MAPE" : np.mean(MAPE),
                    "WMAPE" : np.mean(WMAPE)}