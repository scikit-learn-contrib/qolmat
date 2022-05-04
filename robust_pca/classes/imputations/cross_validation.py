import pandas as pd
import numpy as np
import skopt
from typing import Optional
from sklearn.utils import resample
from math import floor

class CrossValidation:
    def __init__(self, 
                 model, 
                 search_space=None, 
                 search_name=None,
                 value_params=None, 
                 cv=2, 
                 loss_norm="1",
                 ratio_missing=0.1,
                 corruption="missing"
        ):
        self.model = model
        self.search_space = search_space
        self.search_name = search_name
        self.value_params = value_params
        self.cv = cv
        self.loss_norm = loss_norm
        self.ratio_missing = ratio_missing
        self.corruption = corruption
        
        
    def create_corruptions(
                self,
                signal: pd.Series, 
                random_state: Optional[int]=129
                ):
        
        indices = signal.loc[~np.isnan(signal) & signal > 0].index
        self.indices = resample(
                                indices,
                                replace=False,
                                n_samples=floor(len(indices) * self.ratio_missing),
                                random_state=random_state,
                                stratify=None,
                            )           
                    
        self.corrupted_signal = signal.copy()
        if self.corruption == "missing":
            self.corrupted_signal.loc[self.indices] = np.nan
        elif self.corruption == "outlier":
            self.corrupted_signal.loc[self.indices] = np.random.randint(
                0, high=3*np.max(signal), size=(int(len(signal)*self.ratio_missing)))
    
        
    def loss_function(self):
        return np.sum(
                    abs(
                        self.signal[self.indices]
                        - self.imputed[self.indices]
                    )
                )
        
        
    def objective(self, args):
        
        for param_name, param_value in zip(self.search_name, args):
            setattr(self.model, param_name, param_value)
        
        errors = []
        for _ in range(self.cv):
            self.create_corruptions(self.signal)
            self.model.fit(self.corrupted_signal)
            self.imputed = self.model.imputed
            errors.append(self.loss_function())
        
        return np.nanmean(errors)


    def fit(self, signal:pd.Series):
        self.signal = signal
        
        if self.search_space is None:
            self.model.fit(self.signal)
            
        else:
            res = skopt.gp_minimize(
                self.objective,
                self.search_space,
                n_calls=10,
                random_state=42,
                n_jobs=-1,
            )
            
            for param_name, param_value in zip(self.search_name, res.x):
                setattr(self.model, param_name, param_value)
            
            self.model.fit(self.signal)
            
        self.imputed = self.model.imputed
        