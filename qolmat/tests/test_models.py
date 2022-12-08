from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from qolmat.imputations import models


def test_impute_by_mean_fit_transform() -> None:
    test_imputer = models.ImputeByMean()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [2, 2, 4, 4], [3, 3, 9, 9], [2, 2, 2, 2]]
    )


def test_impute_by_median_fit_transform() -> None:
    test_imputer = models.ImputeByMedian()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_by_mode_fit_transform() -> None:
    test_imputer = models.ImputeByMode()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_random_fit_transform() -> None:
    test_imputer = models.ImputeRandom()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]]
        )
    )

    assert set(res.iloc[1, :].values).intersection(
        set([item for sublist in res.iloc[[0, 2, 3], :].values for item in sublist])
    ) == set(res.iloc[1, :].values)


def test_impute_LOCF_fit_transform() -> None:
    test_imputer = models.ImputeLOCF()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_NOCB_fit_transform() -> None:
    test_imputer = models.ImputeNOCB()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [5, 2, 2, 1], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [5, 2, 2, 1], [5, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_linear_interpolation_fit_transform() -> None:
    test_imputer = models.ImputeByInterpolation(method="linear")
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 3, 3], [4, 4, 4, 4]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
    )


def test_impute_linear_interpolation_fit_transform() -> None:
    test_imputer = models.ImputeKNN(k=2)
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 5], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res,
        [[1, 1, 1, 1], [4 / 3, 10 / 6, 10 / 6, 16 / 6], [1, 2, 2, 5], [2, 2, 2, 2]],
    )


def test_impute_linear_interpolation_fit_transform() -> None:
    test_imputer = models.ImputeKNN(k=2)
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 5], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res,
        [[1, 1, 1, 1], [4 / 3, 10 / 6, 10 / 6, 16 / 6], [1, 2, 2, 5], [2, 2, 2, 2]],
    )
