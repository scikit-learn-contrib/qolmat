from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from qolmat.imputations import imputers

df_complete = pd.DataFrame({"col1": [0, 1, 2, 3, 4], "col2": [-1, 0, 0.5, 1, 1.5]})


def test_ImputerMean_sklearn_estim() -> None:
    """Test that ImputerMean is an sklearn estimator"""
    check_estimator(imputers.ImputerMean())


# @parametrize_with_checks([imputers.ImputerMean(),
# imputers.ImputerMedian(), imputers.ImputerMode(), imputers.ImputerShuffle(),
# imputers.ImputerLOCF(), imputers.ImputerNOCB(), imputers.ImputerInterpolation(),
# imputers.ImputerResiduals()])
# def test_sklearn_compatible_estimator(
#     estimator: imputers.Imputer, check: Any
# ) -> None:
#     """Check compatibility with sklearn, using sklearn estimator checks API."""
#     check(estimator)

# @parametrize_with_checks([imputers.ImputerMean()])
# def test_sklearn_compatible_estimator(
#     estimator: imputers.Imputer, check: Any
# ) -> None:
#     """Check compatibility with sklearn, using sklearn estimator checks API."""
#     check(estimator)
