import pytest
import pandas as pd
import numpy as np

from typing import Any
from sklearn.ensemble import ExtraTreesRegressor

from qolmat.imputations import imputers


df_complete = pd.DataFrame({"col1": [0, 1, 2, 3, 4], "col2": [-1, 0, 0.5, 1, 1.5]})

df_incomplete = pd.DataFrame(
    {"col1": [0, np.nan, 2, 3, np.nan], "col2": [-1, np.nan, 0.5, np.nan, 1.5]}
)

df_timeseries = pd.DataFrame(
    pd.DataFrame(
        {
            "col1": [i for i in range(20)],
            "col2": [0, np.nan, 2, np.nan, 2] + [i for i in range(5, 20)],
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


@pytest.mark.parametrize(
    "imputer",
    [
        imputers.ImputerMean(),
        imputers.ImputerMedian(),
        imputers.ImputerMode(),
        imputers.ImputerShuffle(),
        imputers.ImputerLOCF(),
        imputers.ImputerNOCB(),
    ],
)
@pytest.mark.parametrize(
    "df", [pd.DataFrame({"col1": [np.nan, np.nan, np.nan], "col2": [1, 2, 3]})]
)
def test_Imputer_fit_transform_on_nan_column(df: pd.DataFrame, imputer: imputers.Imputer) -> None:
    np.testing.assert_raises(ValueError, imputer.fit_transform, df)


@pytest.mark.parametrize("df", ["string", [1, 2, 3]])
def test_fit_transform_not_on_pandas(df: Any) -> None:
    imputer = imputers.ImputerMean()
    np.testing.assert_raises(ValueError, imputer.fit_transform, df)


@pytest.mark.parametrize("df", [df_groups])
def test_fit_transform_on_grouped(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerMean(groups=["col1"])
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [1, 1, 0, 1],
            "col2": [1, 2, 0, 3],
        }
    )
    print(result)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
@pytest.mark.parametrize("df_oracle", [df_complete])
def test_ImputerOracle_fit_transform(df: pd.DataFrame, df_oracle: pd.DataFrame) -> None:
    imputer = imputers.ImputerOracle(df_oracle)
    result = imputer.fit_transform(df)
    expected = df_oracle
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerMean_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerMean()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": [0, 5 / 3, 2, 3, 5 / 3], "col2": [-1, 1 / 3, 0.5, 1 / 3, 1.5]}
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerMedian_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerMedian()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame({"col1": [0, 2, 2, 3, 2], "col2": [-1, 0.5, 0.5, 0.5, 1.5]})
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerMode_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerMode()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame({"col1": [0, 0, 2, 3, 0], "col2": [-1, -1, 0.5, -1, 1.5]})
    np.testing.assert_allclose(result, expected)


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
    expected = pd.DataFrame({"col1": [0, 3, 2, 3, 0], "col2": [-1, 1.5, 0.5, 1.5, 1.5]})
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerLOCF_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerLOCF()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame({"col1": [0, 0, 2, 3, 3], "col2": [-1, -1, 0.5, 0.5, 1.5]})
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerNOCB_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerNOCB()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame({"col1": [0, 2, 2, 3, 3], "col2": [-1, 0.5, 0.5, 1.5, 1.5]})
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerInterpolation_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerInterpolation()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame({"col1": [0, 1, 2, 3, 3], "col2": [-1, -0.25, 0.5, 1, 1.5]})
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_timeseries])
def test_ImputerResiduals_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerResiduals()
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [i for i in range(20)],
            "col2": [0, 0.9523809523809514, 2, 2.0612244897959204, 2] + [i for i in range(5, 20)],
        },
        index=pd.date_range("2023-04-17", periods=20, freq="D"),
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerKNN_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerKNN(n_neighbors=2)
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0, 1.6666666666666667, 2, 3, 1.4285714285714286],
            "col2": [-1, 0.3333333333333333, 0.5, 0.12499999999999994, 1.5],
        }
    )
    np.testing.assert_allclose(result, expected)


# TODO Imputer MICE, randomness not controled


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerRegressor_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerRegressor(model=ExtraTreesRegressor())
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [0, 1.6666666666666667, 2, 3, 1.6666666666666667],
            "col2": [-1, 0.3333333333333333, 0.5, 0.3333333333333333, 1.5],
        }
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("df", [df_timeseries])
def test_ImputerRPCA_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerRPCA(columnwise=True, period=1, max_iter=100)
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [i for i in range(20)],
            "col2": [0, 5.496290833663846, 2, 5.496290833663846, 2] + [i for i in range(5, 20)],
        }
    )
    np.testing.assert_allclose(result, expected)


# TODO Imputeur EM


@pytest.mark.parametrize("df", [df_timeseries])
def test_ImputerEM_fit_transform(df: pd.DataFrame) -> None:
    imputer = imputers.ImputerEM(random_state=42)
    result = imputer.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [i for i in range(20)],
            "col2": [0, 5.496290833663846, 2, 5.496290833663846, 2] + [i for i in range(5, 20)],
        }
    )
    np.testing.assert_allclose(result, expected)
