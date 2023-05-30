import numpy as np
import pandas as pd
import pytest

from qolmat.benchmark import missing_patterns as mp

df = pd.DataFrame(
    {"col1": [i for i in range(100)] + [np.nan], "col2": [2 * i for i in range(100)] + [np.nan]}
)
list_generators = {
    "geo": mp.GeometricHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42),
    "unif": mp.UniformHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42),
    "multi": mp.MultiMarkovHoleGenerator(n_splits=2, ratio_masked=0.1, random_state=42),
}


@pytest.mark.parametrize(
    "df, generator",
    [(df, list_generators["geo"]), (df, list_generators["unif"]), (df, list_generators["multi"])],
)
def test_SamplerHoleGenerator_split(df: pd.DataFrame, generator: mp._HoleGenerator) -> None:
    mask = generator.split(df)[0]
    col1_holes = mask["col1"].sum()
    col2_holes = mask["col2"].sum()
    expected_col1_holes = 10
    expected_col2_holes = 10
    np.testing.assert_allclose(col1_holes, expected_col1_holes)
    np.testing.assert_allclose(col2_holes, expected_col2_holes)
