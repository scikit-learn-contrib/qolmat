import pandas as pd
import numpy as np
import cross_validation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from collections import defaultdict
from typing import Optional
from math import floor
from skopt.space import Categorical, Real, Integer


class Comparator:
    
    def __init__(self,
                 data,
                 line,
                 ratio_missing,
                 models_to_test,
                 search_params={},
                 corruption="missing"):
        self.df = data[data.index.get_level_values("line")==line] 
        self.ratio_missing = ratio_missing
        self.models_to_test = models_to_test
        self.search_params = search_params
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


    def get_errors(
                self,
                signal_ref: pd.Series, 
                signal_imputed: pd.Series,
        ) -> float:
        rmse = mean_squared_error(signal_ref.loc[self.indices],
                                    signal_imputed.loc[self.indices],
                                    squared = False)
        mae = mean_absolute_error(signal_ref.loc[self.indices],
                                    signal_imputed.loc[self.indices])
        mape = np.mean(np.abs(signal_ref.loc[self.indices] -
                                signal_imputed.loc[self.indices])/
                                np.abs(signal_ref.loc[self.indices]))
        wmape = np.mean(np.abs(signal_ref.loc[self.indices] -
                                signal_imputed.loc[self.indices]))/np.mean(
                                np.abs(signal_ref.loc[self.indices]))
        return {"rmse":rmse, "mae":mae, "mape":mape, "wmape":wmape}


    def compare(self):
        self.results = {}
        for tested_model in self.models_to_test:

            search_space = None
            search_name = None
            if str(type(tested_model).__name__)  in self.search_params.keys():
                search_space = []
                search_name = []
                for name_param,vals_params in self.search_params[str(type(tested_model).__name__)].items():
                    search_name.append(name_param)
                    if vals_params["type"] == "Integer":
                        search_space.append(
                            Integer(
                                low=vals_params["min"], high=vals_params["max"], 
                                name=name_param
                            )
                        )
                    elif vals_params["type"] == "Real":
                        search_space.append(
                            Real(
                                low=vals_params["min"], high=vals_params["max"], 
                                name=name_param
                            )
                        )
                    elif vals_params["type"] == "Categorical":
                        search_space.append(
                            Categorical(
                                categories=vals_params["categories"],
                                name=name_param
                            )
                        )

            errors = defaultdict(list)
            for station in list(self.df.index.get_level_values("station").unique())[10:11]:
                series = self.df[self.df.index.get_level_values("station")==station]["load"]
                self.create_corruptions(series)

                cv = cross_validation.CrossValidation(
                    tested_model, 
                    search_space=search_space,
                    search_name=search_name,
                    ratio_missing=self.ratio_missing,
                    corruption=self.corruption
                    )
                cv.fit(self.corrupted_signal)

                for k,v in self.get_errors(series, cv.imputed).items():
                    errors[k].append(v)
                
            self.results[type(tested_model).__name__] = {k:np.mean(v) for k,v in errors.items()}