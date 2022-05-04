from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet
from pykalman import KalmanFilter
from sklearn.impute import KNNImputer 
import os
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
import sys

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        

class ImputeByMean:
    def __init__(self, ) -> None:
        pass
        
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal.copy()
        self.signal = self.signal.reset_index()

        self.imputed = self.signal.groupby(
            [
                self.signal["station"],
                self.signal["datetime"].dt.dayofweek,
                self.signal["datetime"].dt.round('15min'),
                self.signal["direction"],
             ]
            )['load'].apply(
                lambda x:x.fillna(x.mean())
            )
        self.imputed = self.imputed.to_frame()
        self.imputed = self.imputed.fillna(0)
        self.imputed = self.imputed.set_index(signal.index)
        self.imputed = self.imputed["load"]

    def get_hyperparams(self):
        return {}

class ImputeByMedian:
    def __init__(self, ) -> None:
        pass
        
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal.copy()
        self.signal = self.signal.reset_index()
        self.imputed = self.signal.groupby(
            [
                self.signal["station"],
                self.signal["datetime"].dt.dayofweek,
                self.signal["datetime"].dt.round('15min'),
                self.signal["direction"],
             ]
            )['load'].apply(
                lambda x:x.fillna(x.median())
            )
        self.imputed = self.imputed.to_frame()
        self.imputed = self.imputed.fillna(0)
        self.imputed = self.imputed.set_index(signal.index)
        self.imputed = self.imputed["load"]
        
    def get_hyperparams(self):
        return {}

class ImputeByMode:
    def __init__(self, ) -> None:
        pass
        
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal
        self.imputed = self.signal.fillna(self.signal[self.signal.notnull()].mode()[0])
        
    def get_hyperparams(self):
        return {}
   
        
class RandomImpute:
    def __init__(self, ) -> None:
        pass
    
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal
        self.imputed = self.signal.copy()
        number_missing = self.imputed.isnull().sum()
        obs = self.imputed[self.imputed.notnull()]
        self.imputed.loc[self.imputed.isnull()] = np.random.choice(obs.values, number_missing, replace = True)

    def get_hyperparams(self):
        return {}

class ImputeLOCF:
    def __init__(self, ) -> None:
        pass
    
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal.copy()
        self.signal = self.signal.reset_index()
        self.imputed = self.signal.groupby(
            [
                self.signal["station"],
                self.signal["direction"],
             ]
            )['load'].apply(lambda x: x.ffill())
        self.imputed = self.imputed.to_frame()
        self.imputed = self.imputed.fillna(0)
        self.imputed = self.imputed.set_index(signal.index)
        self.imputed = self.imputed["load"]
        # if index of missing values is 0, impute by the TS median
        self.imputed.fillna(np.nanmedian(self.imputed), inplace=True)

    def get_hyperparams(self):
        return {}
        
        
class ImputeNOCB:
    def __init__(self, ) -> None:
        pass
    
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal.copy()
        self.signal = self.signal.reset_index()
        self.imputed = self.signal.groupby(
            [
                self.signal["station"],
                self.signal["direction"],
             ]
            )['load'].apply(lambda x: x.bfill())
        self.imputed = self.imputed.to_frame()
        self.imputed = self.imputed.fillna(0)
        self.imputed = self.imputed.set_index(signal.index)
        self.imputed = self.imputed["load"]
        # if index of missing values is 0, impute by the TS median
        self.imputed.fillna(np.nanmedian(self.imputed), inplace=True)
    
    def get_hyperparams(self):
        return {}
    

class ImputeProphet:
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)
            
    def fit(self, signal:pd.Series) -> None:
        self.signal = signal.copy()
        
        data = pd.DataFrame()
        data["ds"] = signal.index.get_level_values("datetime")
        data["y"] = signal.values
        
        prophet = Prophet(
                    daily_seasonality=self.daily_seasonality, 
                    weekly_seasonality=self.weekly_seasonality, 
                    yearly_seasonality=self.yearly_seasonality,
                    interval_width=self.interval_width
                    )
        with suppress_stdout_stderr():
            prophet.fit(data)
            
        forecast = prophet.predict(data[["ds"]])["yhat"]
        self.imputed = data["y"].fillna(forecast)
        self.imputed = self.imputed.to_frame("load")
        self.imputed = self.imputed.set_index(signal.index)
        self.imputed = self.imputed["load"]

    def get_hyperparams(self):
        return {
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "interval_width": self.interval_width
        }
        

class ImputeKNN:
    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

        
    def fit(self, signal:pd.Series) -> None:
        # self.signal = signal.reset_index()
        # missing_indices = self.signal[self.signal.isnull()].index
        # for i in missing_indices:
        #     imputer = KNNImputer(n_neighbors=self.k)
        #     self.imputed = imputer.fit_transform( self.signal.loc[:i+1])
        # res = pd.Series([a[0] for a in self.imputed])
        
        self.signal = np.asarray(signal).reshape(-1,1)
        imputer = KNNImputer(n_neighbors=self.k)
        res = imputer.fit_transform(self.signal)
        res = pd.Series([a[0] for a in res], index=signal.index)
        self.imputed = res.fillna(np.nanmedian(res))
        
    def get_hyperparams(self):
        return {"k": self.k}