from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import skopt
from . import utils


class CrossValidation:
    """
    This class implements a cross-validation to find the hyperparameters
    that minimize a reconstruction loss (L1 or L2) over mutliple subsets

    Parameters
    ----------
    model:
    search_space: Optional[Dict[str, Union[int, float, str]]]
        search space for the hyperparameters
    cv: Optional[int ]
        number of splits. By default the value is set to 4
    n_calls: Optional[int]
        number of calls. By default the value is set to 10
    n_jobs: Optional[int]
        The number of parallel jobs to run for neighbors search.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors. By default the value is set to -1
    loss_norm: Optional[int]
        loss norm to evaluate the reconstruction. By default the value is set to 1
    ratio_missing: Optional[float]
        ratio of artificially missing data. By default the value is set to 0.1
    corruption: Optional[str]
        type of corruption: "missing" or "outlier". By default the value is set to "missing"
    verbose: Optional[bool]
        Controls the verbosity: the higher, the more messages. By default the value is set to True
    """

    def __init__(
        self,
        model,
        search_space,
        hole_generator,
        n_calls=10,
        n_jobs=-1,
        loss_norm=1,
        verbose=True,
    ):
        self.model = model
        self.search_space = search_space
        self.hole_generator = hole_generator
        self.n_calls = n_calls
        self.n_jobs = n_jobs
        self.loss_norm = loss_norm
        self.verbose = verbose

    def loss_function(self, initial: pd.DataFrame, imputed: pd.DataFrame) -> float:
        """
        Compute the loss function associated to the loss_norm parameter

        Parameters
        ----------
        initial : pd.DataFrame
            initial dataframe, before creating artificial corruptions
        imputed : pd.DataFrame
            imputed dataframe

        Returns
        -------
        float
            loss

        Raises
        ------
        ValueError
            the loss_norm has to be 1 or 2
        """
        initial_nan = np.where(self.df_is_altered, np.nan, initial)
        imputed_nan = np.where(self.df_is_altered, np.nan, imputed)
        if self.loss_norm == 1:
            return np.nansum(np.abs(initial_nan - imputed_nan))
        elif self.loss_norm == 2:
            return np.sqrt(np.nansum(np.square(initial_nan - imputed_nan)))
        else:
            raise ValueError("loss_norm has to be 0 or 1 (int)")

    def _set_params(self, all_params: Dict[str, Union[float, str]]):
        """
        Set the hyperparameters to the model

        Parameters
        ----------
        all_params : Dict[str, Union[int, float, str]]
            dictionary containing the hyperparameters and their value
        """
        if hasattr(self.model, "rpca"):
            for param_name, param_value in all_params.items():
                setattr(self.model.rpca, param_name, param_value)
        else:
            for param_name, param_value in all_params.items():
                setattr(self.model, param_name, param_value)

    def objective(self):
        """
        Defien the objective function for the cross-validation

        Returns
        -------
        _type_
            objective function
        """

        @skopt.utils.use_named_args(self.search_space)
        def obj_func(**all_params):

            self._set_params(all_params=all_params)
            if self.verbose:
                print(all_params)

            errors = []

            for df_mask in self.hole_generator.split(self.X):
                self.df_is_altered = df_mask
                self.df_corrupted = self.X[df_mask]
                imputed = self.model.fit_transform(self.df_corrupted)
                error = self.loss_function(self.X, imputed)
                errors.append(error)

            mean_errors = np.mean(errors)
            if self.verbose:
                print(mean_errors)
            return mean_errors

        return obj_func

    def fit_transform(
        self, X: pd.DataFrame, return_hyper_params: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        Fit and transform estimator and impute the missing values.

        Parameters
        ----------
        X : pd.DataFrame
            dataframe to impute
        return_hyper_params : Optional[bool], optional
            _description_, by default False

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """
        self.X = X

        res = skopt.gp_minimize(
            self.objective(),
            self.search_space,
            n_calls=self.n_calls,
            n_initial_points=self.n_calls // 5,
            random_state=42,
            n_jobs=self.n_jobs,
        )

        best_params = {
            self.search_space[param].name: res["x"][param]
            for param in range(len(res["x"]))
        }

        self._set_params(all_params=best_params)
        imputed_X = self.model.fit_transform(self.X)

        if return_hyper_params:
            imputed_X = list(imputed_X) + [best_params]
            return imputed_X
        return imputed_X
