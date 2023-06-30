import copy
import logging
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

# import skopt
# from skopt.space import Categorical, Dimension, Integer, Real
import hyperopt as ho
from hyperopt.pyll.base import Apply as hoApply
from qolmat.benchmark import metrics

from qolmat.benchmark.missing_patterns import _HoleGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HyperValue = Union[int, float, str]


def get_objective(imputer, df, generator, metric, names_hyperparams) -> Callable:
    """
    Define the objective function for the cross-validation

    Returns
    -------
    _type_
        objective function
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


def optimize(imputer, df, generator, metric, dict_spaces, max_evals=100):
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
    imputer = copy.deepcopy(imputer)
    if dict_spaces == {}:
        return imputer
    names_hyperparams = list(dict_spaces.keys())
    values_hyperparams = list(dict_spaces.values())
    imputer.imputer_params = tuple(set(imputer.imputer_params) | set(dict_spaces.keys()))
    fun_obj = get_objective(imputer, df, generator, metric, names_hyperparams)
    hyperparams = ho.fmin(
        fn=fun_obj, space=values_hyperparams, algo=ho.tpe.suggest, max_evals=max_evals
    )

    # hyperparams = deflat_hyperparams(hyperparams_flat)
    for key, value in hyperparams.items():
        setattr(imputer, key, value)
    # imputer.hyperparams = hyperparams
    return imputer
