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
def test_Generators_split(df: pd.DataFrame, generator: mp._HoleGenerator) -> None:
    mask = generator.split(df)[0]
    col1_holes = mask["col1"].sum()
    col2_holes = mask["col2"].sum()
    expected_col1_holes = 10
    expected_col2_holes = 10
    np.testing.assert_allclose(col1_holes, expected_col1_holes)
    np.testing.assert_allclose(col2_holes, expected_col2_holes)


df_group = pd.DataFrame(
    {"col1": [i for i in range(100)] + [np.nan], "col2": [2 * i for i in range(100)] + [np.nan]},
    index=np.concatenate([[i for i in range(33)] for j in range(3)] + [[0, 0]]),
)


@pytest.mark.parametrize(
    "df, generator",
    [
        (
            df_group,
            mp.GroupedHoleGenerator(
                n_splits=2, ratio_masked=0.1, random_state=42, groups=["group"]
            ),
        )
    ],
)
def test_GroupedHoleGenerator_split(df: pd.DataFrame, generator: mp._HoleGenerator) -> None:
    df.index = df.index.rename("group")

    mask = generator.split(df)[0]

    num_group = df.index.unique().size
    holes_grouped = mask.groupby("group").sum()
    num_groups_holes_col1 = len(holes_grouped[holes_grouped["col1"] > 1])
    num_groups_holes_col2 = len(holes_grouped[holes_grouped["col2"] > 1])

    ratio_col1 = num_groups_holes_col1 / num_group
    ratio_col2 = num_groups_holes_col2 / num_group

    expected_ratio_col1 = 0.1
    expected_ratio_col2 = 0.1

    np.testing.assert_allclose(ratio_col1, expected_ratio_col1, atol=0.05)
    np.testing.assert_allclose(ratio_col2, expected_ratio_col2, atol=0.05)


df_complet = pd.DataFrame({"col1": [i for i in range(100)], "col2": [2 * i for i in range(100)]})


@pytest.mark.parametrize(
    "df, generator",
    [
        (df_complet, list_generators["geo"]),
        (df_complet, list_generators["unif"]),
        (df_complet, list_generators["multi"]),
    ],
)
def test_SamplerHoleGenerator_without_real_nans(
    df: pd.DataFrame, generator: mp._HoleGenerator
) -> None:
    real_nan = np.random.choice([True, False], size=df.size, p=[0.2, 0.8]).reshape(100, 2)
    df[real_nan] = np.nan

    mask = generator.split(df)[0]

    loc_real_nans_col1 = df.index[df["col1"].isna()]
    loc_mask_col1 = mask.index[mask["col1"]]

    loc_real_nans_col2 = np.where(df["col2"].isna())[0]
    loc_mask_col2 = mask.index[mask["col2"]]

    np.testing.assert_allclose(len(set(loc_real_nans_col1) & set(loc_mask_col1)), 0)
    np.testing.assert_allclose(len(set(loc_real_nans_col2) & set(loc_mask_col2)), 0)
