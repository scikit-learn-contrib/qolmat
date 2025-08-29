from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import (
    parametrize_with_checks,
)

from qolmat.benchmark.hyperparameters import HyperValue
from qolmat.imputations import imputers

df_complete = pd.DataFrame(
    {"col1": [0, 1, 2, 3, 4], "col2": [-1, 0, 0.5, 1, 1.5]}
)

df_incomplete = pd.DataFrame(
    {"col1": [0, np.nan, 2, 3, np.nan], "col2": [-1, np.nan, 0.5, np.nan, 1.5]}
)

df_mixed = pd.DataFrame(
    {
        "col1": [0, np.nan, 2, 3, np.nan],
        "col2": ["a", np.nan, "b", np.nan, "b"],
    }
)

df_timeseries = pd.DataFrame(
    pd.DataFrame(
        {
            "col1": list(range(20)),
            "col2": [0, np.nan, 2, np.nan, 2] + list(range(5, 20)),
        },
        index=pd.date_range("2023-04-17", periods=20, freq="D"),
    )
)

df_groups = pd.DataFrame(
    {
        "col1": [1, 1, 0, 1],
        "col2": [1, np.nan, 0, 3],
    }
)


def test_hyperparameters_get_hyperparameters() -> None:
    imputer = imputers.ImputerKNN(n_neighbors=3)
    hyperparams = imputer.get_hyperparams("col")

    assert hyperparams == {"n_neighbors": 3, "weights": "distance"}


hyperparams_global = {
    "lam/col1": 4.7,
    "lam/col2": 1.5,
    "tolerance": 0.07,
    "max_iterations": 100,
    "norm": "L1",
}

expected1 = {
    "lam": 4.7,
    "tau": None,
    "mu": None,
    "rank": None,
    "list_etas": (),
    "list_periods": (),
    "tolerance": 0.07,
    "norm": "L1",
    "max_iterations": 100,
    "period": 1,
}

expected2 = {
    "lam": 1.5,
    "tau": None,
    "mu": None,
    "rank": None,
    "list_etas": (),
    "list_periods": (),
    "tolerance": 0.07,
    "norm": "L1",
    "max_iterations": 100,
    "period": 1,
}


@pytest.mark.parametrize(
    "col, expected", [("col1", expected1), ("col2", expected2)]
)
def test_hyperparameters_get_hyperparameters_modified(
    col: str, expected: Dict[str, HyperValue]
) -> None:
    imputer = imputers.ImputerRpcaNoisy()
    for key, val in hyperparams_global.items():
        setattr(imputer, key, val)
    imputer.imputer_params = tuple(
        set(imputer.imputer_params) | set(hyperparams_global.keys())
    )
    hyperparams = imputer.get_hyperparams(col)

    assert hyperparams == expected


@pytest.mark.parametrize(
    "imputer",
    [
        imputers.ImputerSimple(),
        imputers.ImputerShuffle(),
        imputers.ImputerLOCF(),
        imputers.ImputerNOCB(),
    ],
)
@pytest.mark.parametrize(
    "df", [pd.DataFrame({"col1": [np.nan, np.nan, np.nan], "col2": [1, 2, 3]})]
)
def test_Imputer_fit_transform_on_nan_column(
    df: pd.DataFrame, imputer: imputers._Imputer
) -> None:
    np.testing.assert_raises(ValueError, imputer.fit_transform, df)


@pytest.mark.parametrize("df", "string")
def test_fit_transform_not_on_pandas(df: Any) -> None:
    imputer = imputers.ImputerSimple()
    np.testing.assert_raises(ValueError, imputer.fit_transform, df)


@pytest.mark.parametrize("df", [df_groups])
def test_fit_transform_on_grouped(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerSimple(groups=("col1",))
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [1, 1, 0, 1],
            "col2": [1.0, 2.0, 0.0, 3.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
@pytest.mark.parametrize("df_oracle", [df_complete])
def test_ImputerOracle_fit_transform(
    df: pd.DataFrame, df_oracle: pd.DataFrame
) -> None:
    imputer = imputers.ImputerOracle()
    imputer.set_solution(df_oracle)
    result = imputer.fit_transform(df)
    expected = df_oracle
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_mixed])
def test_ImputerSimple_mean_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerSimple(strategy="mean")
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0, 5 / 3, 2, 3, 5 / 3], "col2": ["a", "b", "b", "b", "b"]}
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("df", [df_mixed])
def test_ImputerSimple_median_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerSimple()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0.0, 2.0, 2.0, 3.0, 2.0], "col2": ["a", "b", "b", "b", "b"]}
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("df", [df_mixed])
def test_ImputerSimple_mode_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerSimple(strategy="most_frequent")
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0.0, 0.0, 2.0, 3.0, 0.0], "col2": ["a", "b", "b", "b", "b"]}
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("df", [pd.DataFrame({"col1": [1, 1, np.nan]})])
def test_ImputerShuffle_fit_transform1(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerShuffle()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame({"col1": [1, 1, 1]})
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerShuffle_fit_transform2(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerShuffle(random_state=42)
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0, 3, 2, 3, 0], "col2": [-1, 1.5, 0.5, 1.5, 1.5]}
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerLOCF_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerLOCF()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0, 0, 2, 3, 3], "col2": [-1, -1, 0.5, 0.5, 1.5]}
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerNOCB_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerNOCB()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0, 2, 2, 3, 3], "col2": [-1, 0.5, 0.5, 1.5, 1.5]}
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerInterpolation_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerInterpolation()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0, 1, 2, 3, 3], "col2": [-1, -0.25, 0.5, 1, 1.5]}
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_timeseries])
def test_ImputerResiduals_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerResiduals(period=7)
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": list(range(20)),
            "col2": [0, 0.953, 2, 2.061, 2] + list(range(5, 20)),
        },
        index=pd.date_range("2023-04-17", periods=20, freq="D"),
    )
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerKNN_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerKNN(n_neighbors=2)
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0, 5 / 3, 2, 3, 1.4285714285714286],
            "col2": [-1, 1 / 3, 0.5, 1 / 8, 1.5],
        }
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerMICE_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerMICE(
        estimator=ExtraTreesRegressor(),
        random_state=42,
        sample_posterior=False,
        max_iter=100,
    )
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0, 3, 2, 3, 3],
            "col2": [-1, 1.5, 0.5, 1.5, 1.5],
        }
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerRegressor_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerRegressor(estimator=ExtraTreesRegressor())
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0, 2, 2, 3, 2],
            "col2": [-1, 0.5, 0.5, 0.5, 1.5],
        }
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_timeseries])
def test_ImputerRpcaNoisy_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerRpcaNoisy(
        columnwise=False, max_iterations=100, tau=1, lam=0.3
    )
    df_omega = df.notna()
    df_result = imputer.fit_transform(df)
    np.testing.assert_allclose(df_result[df_omega], df[df_omega])
    assert df_result.notna().all().all()


index_grouped = pd.MultiIndex.from_product(
    [["a", "b"], range(4)], names=["group", "date"]
)
dict_values = {
    "col1": [0, np.nan, 0, np.nan, 1, 1, 1, 1],
    "col2": [1, 1, 1, 1, 2, 2, 2, 2],
}
df_grouped = pd.DataFrame(dict_values, index=index_grouped)

list_imputers = [
    imputers.ImputerSimple(groups=("group",)),
    imputers.ImputerShuffle(groups=("group",)),
    imputers.ImputerLOCF(groups=("group",)),
    imputers.ImputerNOCB(groups=("group",)),
    imputers.ImputerInterpolation(groups=("group",)),
    imputers.ImputerResiduals(groups=("group",), period=2),
    imputers.ImputerKNN(groups=("group",)),
    imputers.ImputerMICE(groups=("group",)),
    imputers.ImputerRegressor(groups=("group",), estimator=LinearRegression()),
    imputers.ImputerRpcaPcp(groups=("group",)),
    imputers.ImputerRpcaNoisy(groups=("group",)),
    imputers.ImputerSoftImpute(groups=("group",)),
    imputers.ImputerEM(groups=("group",), method="mle"),
]


@pytest.mark.parametrize("imputer", list_imputers)
def test_models_fit_transform_grouped(imputer):
    result = imputer.fit_transform(df_grouped)
    expected = df_grouped.fillna(0)
    np.testing.assert_allclose(result, expected)


@parametrize_with_checks(
    [
        imputers._Imputer(),
        imputers.ImputerOracle(),
        imputers.ImputerSimple(),
        imputers.ImputerShuffle(),
        imputers.ImputerLOCF(),
        imputers.ImputerNOCB(),
        imputers.ImputerInterpolation(),
        imputers.ImputerResiduals(period=2),
        imputers.KNNImputer(),
        imputers.ImputerMICE(),
        imputers.ImputerRegressor(estimator=LinearRegression()),
        imputers.ImputerRpcaNoisy(tau=1, lam=1),
        imputers.ImputerRpcaPcp(lam=1),
        imputers.ImputerSoftImpute(),
        imputers.ImputerEM(),
    ]
)
def test_sklearn_compatible_estimator(
    estimator: imputers._Imputer, check: Any
) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    check(estimator)
