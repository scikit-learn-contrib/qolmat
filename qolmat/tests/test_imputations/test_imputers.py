import pytest
import pandas as pd
import numpy as np

from qolmat.imputations import imputers


df_complete = pd.DataFrame({"col1": [0, 1, 2, 3, 4], "col2": [-1, 0, 0.5, 1, 1.5]})

df_incomplete = pd.DataFrame(
    {"col1": [0, np.nan, 2, 3, np.nan], "col2": [-1, np.nan, 0.5, np.nan, 1.5]}
)


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
    print(result)
    expected = pd.DataFrame({"col1": [0, 2, 2, 3, 2], "col2": [-1, 0.5, 0.5, 1.5, 1.5]})
    np.testing.assert_allclose(result, expected)
