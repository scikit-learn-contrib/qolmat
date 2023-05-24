import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import missing_patterns as mp

df = pd.DataFrame(
    {"col1": [i for i in range(100)] + [np.nan], "col2": [2 * i for i in range(100)] + [np.nan]}
)


@pytest.mark.parametrize("df", [df])
def test_SamplerHoleGenerator_split(df: pd.DataFrame) -> None:
    generator = mp.GeometricHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42)
    mask = generator.split(df)[0]
    col1_holes = sum(mask["col1"])
    col2_holes = sum(mask["col2"])
    expected_col1_holes = 10
    expected_col2_holes = 10
    np.testing.assert_allclose(col1_holes, expected_col1_holes)
    np.testing.assert_allclose(col2_holes, expected_col2_holes)


@pytest.mark.parametrize("df", [df])
def test_UniformHoleGenerator_split(df: pd.DataFrame) -> None:
    generator = mp.UniformHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42)
    mask = generator.split(df)[0]
    col1_holes = sum(mask["col1"])
    col2_holes = sum(mask["col2"])
    expected_col1_holes = 10
    expected_col2_holes = 10
    np.testing.assert_allclose(col1_holes, expected_col1_holes)
    np.testing.assert_allclose(col2_holes, expected_col2_holes)


@pytest.mark.parametrize("df", [df])
def test_MultiMarkovHoleGenerator_split(df: pd.DataFrame) -> None:
    generator = mp.MultiMarkovHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42)
    mask = generator.split(df)[0]
    col1_holes = mask["col1"].sum()
    col2_holes = mask["col2"].sum()
    expected_col1_holes = 10
    expected_col2_holes = 10
    np.testing.assert_allclose(col1_holes, expected_col1_holes)
    np.testing.assert_allclose(col2_holes, expected_col2_holes)
