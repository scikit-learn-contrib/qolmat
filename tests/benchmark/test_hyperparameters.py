from typing import List, Optional, Tuple

import hyperopt as ho
import numpy as np
import pandas as pd

from qolmat.benchmark import hyperparameters

# from hyperparameters import HyperValue
from qolmat.benchmark.missing_patterns import (
    EmpiricalHoleGenerator,
    _HoleGenerator,
)
from qolmat.imputations.imputers import ImputerRpcaNoisy, _Imputer
from qolmat.utils.utils import RandomSetting

df_origin = pd.DataFrame(
    {"col1": [0, np.nan, 2, 4, np.nan], "col2": [-1, np.nan, 0.5, 1, 1.5]}
)
df_imputed = pd.DataFrame(
    {"col1": [0, 1, 2, 3.5, 4], "col2": [-1.5, 0, 1.5, 2, 1.5]}
)
df_mask = pd.DataFrame(
    {
        "col1": [False, False, True, False, False],
        "col2": [True, False, True, True, False],
    }
)
df_corrupted = df_origin.copy()
df_corrupted[df_mask] = np.nan

imputer_rpca = ImputerRpcaNoisy(
    tau=2, random_state=42, columnwise=True, period=1
)
dict_imputers_rpca = {"rpca": imputer_rpca}
generator_holes = EmpiricalHoleGenerator(n_splits=1, ratio_masked=0.5)
dict_config_opti = {
    "rpca": {
        "lam": {
            "col1": {"min": 0.1, "max": 6, "type": "Real"},
            "col2": {"min": 1, "max": 4, "type": "Real"},
        },
        "tolerance": {"min": 1e-6, "max": 0.1, "type": "Real"},
        "max_iter": {"min": 99, "max": 100, "type": "Integer"},
        "norm": {"categories": ["L1", "L2"], "type": "Categorical"},
    }
}


class ImputerTest(_Imputer):
    """Group tests for Imputer."""

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        random_state: RandomSetting = None,
        value: float = 0,
    ) -> None:
        """Init function."""
        super().__init__(
            groups=groups, columnwise=True, random_state=random_state
        )
        self.value = value

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ):
        df_out = df.copy()
        df_out = df_out.fillna(self.value)
        return df_out


class HoleGeneratorTest(_HoleGenerator):
    """Group tests for HoleGenerator."""

    def __init__(self, mask: pd.Series, subset: Optional[List[str]] = None):
        """Init HoleGenerator."""
        super().__init__(n_splits=1, subset=subset)
        self.mask = mask

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate mask."""
        df_out = X.copy()
        for col in df_out:
            df_out[col] = self.mask
        return df_out


def test_hyperparameters_get_objective() -> None:
    """Test get_objective."""
    imputer = ImputerTest()
    generator = HoleGeneratorTest(
        pd.Series([False, False, True, True]), subset=["some_col"]
    )
    metric = "mse"
    names_hyperparams = ["value"]
    df = pd.DataFrame({"some_col": [np.nan, 0, 3, 5]})
    fun_obj = hyperparameters.get_objective(
        imputer, df, generator, metric, names_hyperparams
    )
    assert fun_obj([4]) == 1
    assert fun_obj([0]) == (3**2 + 5**2) / 2


def test_hyperparameters_optimize():
    """Test optimize."""
    imputer = ImputerTest()
    generator = HoleGeneratorTest(
        pd.Series([False, False, True, True]), subset=["some_col"]
    )
    metric = "mse"
    dict_config_opti = {"value": ho.hp.uniform("value", 0, 10)}
    df = pd.DataFrame({"some_col": [np.nan, 0, 3, 5]})
    imputer_opti = hyperparameters.optimize(
        imputer, df, generator, metric, dict_config_opti, max_evals=500
    )
    assert isinstance(imputer_opti, ImputerTest)
    np.testing.assert_almost_equal(imputer_opti.value, 4, decimal=1)
