import gc
import datetime
from math import floor
import signal
from typing import Optional, Union

import skopt
from skopt import gp_minimize
from sklearn.metrics import make_scorer

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state, resample

from .evaluate_imputer import evaluate_imputor
from .rpca import RPCA


class get_objective:

    def __init__(self,
                 imputor: RPCA,
                 imputor_eval: evaluate_imputor,
                 space,
                 scoring):
        self.imputor = imputor
        self.imputor_eval = imputor_eval
        self.space = space
        self.scoring = scoring

    def fit_transform_and_evaluate(self):
        @skopt.utils.use_named_args(self.space)
        def obj_func(**all_params):
            imputor = self.imputor.set_params(**all_params)
            scores = self.imputor_eval.scores_imputor(imputor)
            gc.collect()
            return -scores[self.scoring]

        return obj_func

def rpca_optimizer(imputor: RPCA,
                   imputor_eval: evaluate_imputor,
                   space,
                   scoring,
                   n_random_starts = 10, 
                   epoch = 100,
                   n_jobs = 1,
                   verbose = False):

    objective = get_objective(imputor=imputor,
                              imputor_eval=imputor_eval,
                              space=space,
                              scoring=scoring
    )
    objective_func = objective.fit_transform_and_evaluate()

    print("beging_time = " + str(datetime.now().strftime("%H:%M:%S")))
    result = gp_minimize(
            func=objective_func,
            dimensions=space,
            acq_optimizer="lbfgs",
            random_state=imputor_eval.random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            n_random_starts=n_random_starts,
            n_calls=epoch,
        )
    best_params = {space[index].name: result["x"][index] for index in range(len(result["x"]))}
    best_score = result["fun"]

    print("end_time = " + str(datetime.now().strftime("%H:%M:%S")))
    best_imputor = imputor.set_params(**best_params)
    gc.collect()
    transform_signal = imputor.fit_transform(imputor_eval.signal)
    return best_imputor, best_score, best_params, transform_signal

