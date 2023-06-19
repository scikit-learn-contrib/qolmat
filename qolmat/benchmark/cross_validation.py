import logging
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import skopt
from skopt.space import Categorical, Dimension, Integer, Real

from qolmat.benchmark.missing_patterns import _HoleGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_dimension(dict_bounds: Dict, name_dimension: str) -> Dimension:
    """Get the dimension of hyperparamaters with skopt

    Parameters
    ----------
    dict_bounds : Dict
        Dictionnay of bounds of hyperparameters
    name_dimension : str
        Name of hyperparameters

    Returns
    -------
    Dimension
        In the case Integer, we have a skopt.space.Integer,
        for Real we have skopt.space.Real and
        for Categorical we have skopt.space.Categorical
    """
    if dict_bounds["type"] == "Integer":
        return Integer(low=dict_bounds["min"], high=dict_bounds["max"], name=name_dimension)
    elif dict_bounds["type"] == "Real":
        return Real(low=dict_bounds["min"], high=dict_bounds["max"], name=name_dimension)
    elif dict_bounds["type"] == "Categorical":
        return Categorical(categories=dict_bounds["categories"], name=name_dimension)
    else:
        ValueError("The 'type' must be 'Integer', 'Real' or 'Categorical")


def get_search_space(dict_config_opti_imputer: Dict) -> List[Dimension]:
    """Construct the search space for the tested_model
    based on the dict_config_opti_imputer

    Parameters
    ----------
    dict_config_opti_imputer : Dict

    Returns
    -------
    List[Dimension]
        search space

    """
    list_spaces = []

    for name_hyperparam, value in dict_config_opti_imputer.items():
        # space common for all columns
        if "type" in value:
            list_spaces.append(get_dimension(value, name_hyperparam))
        else:
            for col, dict_bounds in value.items():
                name = f"{name_hyperparam}/{col}"
                list_spaces.append(get_dimension(dict_bounds, name))

    return list_spaces


def deflat_hyperparams(
    hyperparams_flat: Dict[str, Union[float, int, str]]
) -> Dict[str, Union[float, int, str, Dict[str, Union[float, int, str]]]]:
    """
    Set the hyperparameters to the model

    Parameters
    ----------
    hyperparams_flat : Dict[str, Union[int, float, str]]
        dictionary containing the hyperparameters and their value`

    Return
    -------
    Dict
    Deflat hyperparams_flat
    """

    hyperparams: Dict[str, Any] = {}
    for name_dimension, hyperparam in hyperparams_flat.items():
        if "/" not in name_dimension:
            hyperparams[name_dimension] = hyperparam
        else:
            name_hyperparam, col = name_dimension.split("/")
            if name_hyperparam in hyperparams:
                hyperparams[name_hyperparam][col] = hyperparam
            else:
                new_dict: Dict[str, Union[float, int, str]] = {col: hyperparam}
                hyperparams[name_hyperparam] = new_dict
    return hyperparams


class CrossValidation:
    """
    This class implements a cross-validation to find the hyperparameters
    that minimize a reconstruction loss (L1 or L2) over mutliple subsets

    Parameters
    ----------
    imputer: Any
        Imputer with the hyperparameters
    dict_config_opti_imputer: Optional[Dict[str, Union[int, float, str]]]
        search space for the hyperparameters
    hole_generator: _HoleGenerator
        The generator of hole
    n_calls: Optional[int]
        number of calls. By default the value is set to 10
    n_jobs: Optional[int]
        The number of parallel jobs to run for neighbors search.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors. By default the value is set to -1
    loss_norm: Optional[int]
        loss norm to evaluate the reconstruction. By default the value is set to 1
    """

    def __init__(
        self,
        imputer: Any,
        dict_config_opti_imputer: Dict[str, Any],
        hole_generator: _HoleGenerator,
        n_calls: int = 10,
        n_jobs: int = -1,
        loss_norm: int = 1,
    ):
        self.imputer = imputer
        self.dict_config_opti_imputer = dict_config_opti_imputer
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

    def objective(self, df: pd.DataFrame, list_spaces: List[Dimension]) -> Callable:
        """
        Define the objective function for the cross-validation

        Returns
        -------
        _type_
            objective function
        """

        @skopt.utils.use_named_args(list_spaces)
        def obj_func(**hyperparams_flat):
            self.imputer.hyperparams_optim = deflat_hyperparams(hyperparams_flat)

            errors = []

            for df_mask in self.hole_generator.split(df):
                df_origin = df.copy()
                df_corrupted = df_origin.copy()
                df_corrupted[df_mask] = np.nan
                cols_with_nans = df.columns[df.isna().any(axis=0)].tolist()
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

    def optimize_hyperparams(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize hyperparamaters

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame masked

        Returns
        -------
        Dict[str, Any]
            hyperparameters optimize flat
        """
        list_spaces = get_search_space(self.dict_config_opti_imputer)
        res = skopt.gp_minimize(
            self.objective(df, list_spaces),
            dimensions=list_spaces,
            n_calls=self.n_calls,
            n_initial_points=max(5, self.n_calls // 5),
            random_state=self.imputer.random_state,
            n_jobs=self.n_jobs,
        )

        hyperparams_flat = {space.name: val for space, val in zip(list_spaces, res["x"])}
        hyperparams = deflat_hyperparams(hyperparams_flat)
        return hyperparams
