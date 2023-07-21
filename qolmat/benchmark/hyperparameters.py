import copy
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

# import skopt
# from skopt.space import Categorical, Dimension, Integer, Real
import hyperopt as ho
from hyperopt.pyll.base import Apply as hoApply
from qolmat.benchmark import metrics

from qolmat.benchmark.missing_patterns import _HoleGenerator
from qolmat.imputations.imputers import _Imputer
from qolmat.utils.utils import HyperValue


def get_objective(
    imputer: _Imputer,
    df: pd.DataFrame,
    generator: _HoleGenerator,
    metric: str,
    names_hyperparams: List[str],
) -> Callable:
    """
    Define the objective function, which is the average metric computed over the folds provided by
    the hole generator, using a cross-validation.

    Parameters
    ----------
    imputer: _Imputer
        Imputer that should be optimized, it should at least have a fit_transform method and an
        imputer_params attribute
    generator: _HoleGenerator
        Generator creating the masked values in the nested cross validation allowing to measure the
         imputer performance
    metric: str
        Metric used as perfomance indicator, common values are `mse` and `mae`
    names_hyperparams: List[str]
        List of the names of the hyperparameters which are being optimized

    Returns
    -------
    Callable[List[HyperValue], float]
        Objective function
    """

    def fun_obf(args: List[HyperValue]) -> float:
        for key, value in zip(names_hyperparams, args):
            setattr(imputer, key, value)

        list_errors = []

        for df_mask in generator.split(df):
            df_origin = df.copy()
            df_corrupted = df_origin.copy()
            df_corrupted[df_mask] = np.nan
            df_imputed = imputer.fit_transform(df_corrupted)
            subset = generator.subset
            fun_metric = metrics.get_metric(metric)
            errors = fun_metric(df_origin[subset], df_imputed[subset], df_mask[subset])
            list_errors.append(errors)

        mean_errors = np.mean(errors)
        return mean_errors

    return fun_obf


def optimize(
    imputer: _Imputer,
    df: pd.DataFrame,
    generator: _HoleGenerator,
    metric: str,
    dict_config: Dict[str, HyperValue],
    max_evals: int = 100,
    verbose: bool = False,
):
    """Return the provided imputer with hyperparameters optimized in the provided range in order to
     minimize the provided metric.

    Parameters
    ----------
    imputer: _Imputer
        Imputer that should be optimized, it should at least have a fit_transform method and an
        imputer_params attribute
    generator: _HoleGenerator
        Generator creating the masked values in the nested cross validation allowing to measure the
         imputer performance
    metric: str
        Metric used as perfomance indicator, common values are `mse` and `mae`
    dict_config: Dict[str, HyperValue]
        Search space for the tested hyperparameters
    max_evals: int
        Maximum number of evaluation of the performance of the algorithm. Each estimation involves
        one call to fit_transform per fold returned by the generator. See the n_fold attribute.
    verbose: bool
        Verbosity switch, usefull for imputers that can have unstable behavior for some
        hyperparameters values

    Returns
    -------
    _Imputer
        Optimized imputer
    """
    imputer = copy.deepcopy(imputer)
    if dict_config == {}:
        return imputer
    names_hyperparams = list(dict_config.keys())
    values_hyperparams = list(dict_config.values())
    imputer.imputer_params = tuple(set(imputer.imputer_params) | set(dict_config.keys()))
    if verbose and hasattr(imputer, "verbose"):
        setattr(imputer, "verbose", False)
    fun_obj = get_objective(imputer, df, generator, metric, names_hyperparams)
    hyperparams = ho.fmin(
        fn=fun_obj, space=values_hyperparams, algo=ho.tpe.suggest, max_evals=max_evals
    )

    for key, value in hyperparams.items():
        setattr(imputer, key, value)
    return imputer
