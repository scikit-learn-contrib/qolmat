import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import skopt
from skopt.space import Dimension

from qolmat.benchmark.missing_patterns import _HoleGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CrossValidation:
    """
    This class implements a cross-validation to find the hyperparameters
    that minimize a reconstruction loss (L1 or L2) over mutliple subsets

    Parameters
    ----------
    model:
    search_space: Optional[Dict[str, Union[int, float, str]]]
        search space for the hyperparameters
    hole_generator:

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
    """

    def __init__(
        self,
        imputer: any,
        list_spaces: List[Dimension],
        hole_generator: _HoleGenerator,
        n_calls: int = 10,
        n_jobs: int = -1,
        loss_norm: int = 1,
    ):
        self.imputer = imputer
        self.list_spaces = list_spaces
        self.hole_generator = hole_generator
        self.n_calls = n_calls
        self.n_jobs = n_jobs
        self.loss_norm = loss_norm

    def loss_function(
        self,
        df_origin: pd.DataFrame,
        df_imputed: pd.DataFrame,
        df_mask: pd.DataFrame,
    ) -> float:
        """
        Compute the loss function associated to the loss_norm parameter

        Parameters
        ----------
        df_origin : pd.DataFrame
            initial dataframe, before creating artificial corruptions
        df_imputed : pd.DataFrame
            imputed dataframe
        df_mask : pd.DataFrame
            boolean dataframe, True where artificial corruptions are created

        Returns
        -------
        float
            loss

        Raises
        ------
        ValueError
            the loss_norm has to be 1 or 2
        """
        if self.loss_norm == 1:
            return np.nansum(np.abs(df_origin[df_mask] - df_imputed[df_mask]))
        elif self.loss_norm == 2:
            return np.sqrt(np.nansum(np.square(df_origin[df_mask] - df_imputed[df_mask])))
        else:
            raise ValueError("loss_norm has to be 0 or 1 (int)")

    def deflat_hyperparams(self, hyperparams_flat: Dict[str, Union[float, int, str]]) -> Dict:
        """
        Set the hyperparameters to the model

        Parameters
        ----------
        hyperparams_flat : Dict[str, Union[int, float, str]]
            dictionary containing the hyperparameters and their value
        """
        hyperparams = {}
        for name_dimension, hyperparam in hyperparams_flat.items():
            if "/" not in name_dimension:
                hyperparams[name_dimension] = hyperparam
            else:
                name_hyperparam, col = name_dimension.split("/")
                if name_hyperparam in hyperparams:
                    hyperparams[name_hyperparam][col] = hyperparam
                else:
                    hyperparams[name_hyperparam] = {col: hyperparam}
        return hyperparams

    def objective(self, X):
        """
        Define the objective function for the cross-validation

        Returns
        -------
        _type_
            objective function
        """

        @skopt.utils.use_named_args(self.list_spaces)
        def obj_func(**hyperparams_flat):
            self.imputer.hyperparams_optim = self.deflat_hyperparams(hyperparams_flat)

            errors = []

            for df_mask in self.hole_generator.split(X):
                df_origin = X.copy()
                df_corrupted = df_origin.copy()
                df_corrupted[df_mask] = np.nan
                cols_with_nans = X.columns[X.isna().any(axis=0)].tolist()
                imputed = self.imputer.fit_transform(df_corrupted)

                error = self.loss_function(
                    df_origin.loc[:, cols_with_nans],
                    imputed.loc[:, cols_with_nans],
                    df_mask.loc[:, cols_with_nans],
                )
                errors.append(error)

            mean_errors = np.mean(errors)
            return mean_errors

        return obj_func

    def fit_transform(
        self, df: pd.DataFrame, return_hyper_params: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        Fit and transform estimator and impute the missing values.

        Parameters
        ----------
        X : pd.DataFrame
            dataframe to impute
        return_hyper_params : Optional[bool]
            by default False

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """

        n0 = max(5, self.n_calls // 5)
        print("---")
        print(self.n_calls)
        print(n0)

        # res = skopt.gp_minimize(
        #     self.objective(X=df),
        #     dimensions=self.list_spaces,
        #     n_calls=self.n_calls,
        #     n_initial_points=n0,
        #     random_state=42,
        #     n_jobs=self.n_jobs,
        # )

        res = skopt.gp_minimize(
            self.objective(X=df),
            dimensions=self.list_spaces,
            n_calls=self.n_calls,
            n_initial_points=n0,
            random_state=42,
            n_jobs=self.n_jobs,
        )

        hyperparams_flat = {space.name: val for space, val in zip(self.list_spaces, res["x"])}
        print(f"Optimal hyperparameters : {hyperparams_flat}")
        print(f"Results: {res}")

        self.imputer.hyperparams_optim = self.deflat_hyperparams(hyperparams_flat)
        df_imputed = self.imputer.fit_transform(df)

        if return_hyper_params:
            return df_imputed, hyperparams_flat
        return df_imputed
