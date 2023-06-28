from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import hyperparameters
from qolmat.benchmark.hyperparameters import HyperValue

# from hyperparameters import HyperValue
from qolmat.benchmark.missing_patterns import _HoleGenerator, EmpiricalHoleGenerator
from qolmat.imputations.imputers import Imputer, ImputerRPCA

import hyperopt as ho

df_origin = pd.DataFrame({"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]})
df_imputed = pd.DataFrame({"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]})
df_mask = pd.DataFrame(
    {"col1": [False, False, True, False, False], "col2": [True, False, True, True, False]}
)
df_corrupted = df_origin.copy()
df_corrupted[df_mask] = np.nan

imputer_rpca = ImputerRPCA(tau=2, random_state=42, columnwise=True, period=1)
dict_imputers_rpca = {"rpca": imputer_rpca}
generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
dict_config_opti = {
    "rpca": {
        "lam": {
            "col1": {"min": 0.1, "max": 6, "type": "Real"},
            "col2": {"min": 1, "max": 4, "type": "Real"},
        },
        "tol": {"min": 1e-6, "max": 0.1, "type": "Real"},
        "max_iter": {"min": 99, "max": 100, "type": "Integer"},
        "norm": {"categories": ["L1", "L2"], "type": "Categorical"},
    }
}
dict_config_opti_imputer = dict_config_opti["rpca"]
hyperparams_global = {"lam/col1": 4.7, "lam/col2": 1.5, "tol": 0.07, "max_iter": 100, "norm": "L1"}

result_params_expected = {
    "lam1": (0.1, 6),
    "lam2": (1, 4),
    "tol": (1e-6, 0.1),
    "max_iter": (99, 100),
    "norm": ("L1", "L2"),
}

expected1 = {
    "lam": 4.7,
    "tol": 0.07,
    "max_iter": 100,
    "norm": "L1",
}

expected2 = {
    "lam": 1.5,
    "tol": 0.07,
    "max_iter": 100,
    "norm": "L1",
}


@pytest.mark.parametrize("hyperparams_global", [hyperparams_global])
@pytest.mark.parametrize("col, expected", [("col1", expected1), ("col2", expected2)])
def test_hyperparameters_get_hyperparameters(
    hyperparams_global: Dict[str, HyperValue], col: str, expected: Dict[str, HyperValue]
) -> None:
    hyperparams = hyperparameters.get_hyperparams(hyperparams_global, col)

    assert hyperparams == expected


class ImputerTest(Imputer):
    def __init__(
        self,
        groups: List[str] = [],
        random_state: Union[None, int, np.random.RandomState] = None,
        value: float = 0,
    ) -> None:
        super().__init__(groups=groups, columnwise=True, random_state=random_state)
        self.value = value

    def fit_transform_element(self, df: pd.DataFrame):
        df_out = df.copy()
        df_out = df_out.fillna(self.value)
        return df_out


class HoleGeneratorTest(_HoleGenerator):
    def __init__(self, mask: pd.Series, subset: Optional[List[str]] = None):
        super().__init__(n_splits=1, subset=subset)
        self.mask = mask

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out = X.copy()
        for col in df_out:
            df_out[col] = self.mask
        return df_out


def test_hyperparameters_get_objective() -> None:
    imputer = ImputerTest()
    generator = HoleGeneratorTest(pd.Series([False, False, True, True]), subset=["some_col"])
    metric = "mse"
    names_hyperparams = ["value"]
    df = pd.DataFrame({"some_col": [np.nan, 0, 3, 5]})
    fun_obj = hyperparameters.get_objective(imputer, df, generator, metric, names_hyperparams)
    assert fun_obj([4]) == 1
    assert fun_obj([0]) == (3**2 + 5**2) / 2


def test_hyperparameters_optimize():
    imputer = ImputerTest()
    generator = HoleGeneratorTest(pd.Series([False, False, True, True]), subset=["some_col"])
    metric = "mse"
    dict_config_opti = {"value": ho.hp.uniform("value", 0, 10)}
    df = pd.DataFrame({"some_col": [np.nan, 0, 3, 5]})
    imputer_opti = hyperparameters.optimize(
        imputer, df, generator, metric, dict_config_opti, max_evals=200
    )
    assert isinstance(imputer_opti, ImputerTest)
    np.testing.assert_almost_equal(imputer_opti.value, 4, decimal=2)
