from __future__ import annotations

import numpy as np
import pandas as pd

from qolmat.benchmark import utils
from qolmat.imputations import models
from qolmat.benchmark import missing_patterns


def test_impute_by_mean_fit_transform() -> None:
    test_imputer = models.ImputeByMean()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [3, 3, 9, 9],
                [2, 2, 2, 3],
            ]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [2, 2, 4, 4], [3, 3, 9, 9], [2, 2, 2, 3]]
    )


def test_impute_by_median_fit_transform() -> None:
    test_imputer = models.ImputeByMedian()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [1, 2, 2, 1],
                [2, 2, 2, 2],
            ]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_by_mode_fit_transform() -> None:
    test_imputer = models.ImputeByMode()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [1, 2, 2, 1],
                [2, 2, 2, 2],
            ]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_random_fit_transform() -> None:
    test_imputer = models.ImputeRandom()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [1, 2, 2, 1],
                [2, 2, 2, 2],
            ]
        )
    )

    assert set(res.iloc[1, :].values).intersection(
        set(
            [
                item
                for sublist in res.iloc[[0, 2, 3], :].values
                for item in sublist
            ]
        )
    ) == set(res.iloc[1, :].values)


def test_impute_LOCF_fit_transform() -> None:
    test_imputer = models.ImputeLOCF()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [1, 2, 2, 1],
                [2, 2, 2, 2],
            ]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_NOCB_fit_transform() -> None:
    test_imputer = models.ImputeNOCB()
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [5, 2, 2, 1],
                [2, 2, 2, 2],
            ]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [5, 2, 2, 1], [5, 2, 2, 1], [2, 2, 2, 2]]
    )


def test_impute_linear_interpolation_fit_transform() -> None:
    test_imputer = models.ImputeByInterpolation(method='linear')
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
    )


def test_impute_KNN_fit_transform() -> None:
    test_imputer = models.ImputeKNN(k=2)
    res = test_imputer.fit_transform(
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [np.nan, np.nan, np.nan, np.nan],
                [1, 2, 2, 5],
                [2, 2, 2, 2],
            ]
        )
    )

    np.testing.assert_array_equal(
        res,
        [
            [1, 1, 1, 1],
            [4 / 3, 10 / 6, 10 / 6, 16 / 6],
            [1, 2, 2, 5],
            [2, 2, 2, 2],
        ],
    )


def test_impute_KNN_hyperparameters() -> None:
    test_imputer = models.ImputeKNN(k=2)

    assert len(test_imputer.get_hyperparams()) == 1
    assert test_imputer.get_hyperparams() == {'k': 2}


def test_impute_MICE_fit_transform() -> None:
    from sklearn.ensemble import ExtraTreesRegressor

    n, p = 100, 4
    mu = np.random.rand(p)
    A = np.random.rand(p, p)
    sigma = np.dot(A, A.transpose())
    df_init = np.random.multivariate_normal(mu, sigma, n)
    df_init = pd.DataFrame(data=df_init)

    df_corrupted = df_init.copy()
    X_miss_mcar = missing_patterns.produce_NA(
        df_corrupted, p_miss=0.2, mecha='MCAR'
    )
    df_corrupted = X_miss_mcar['X_incomp']
    mask_mcar = X_miss_mcar['mask']

    test_imputer = models.ImputeMICE(
        estimator=ExtraTreesRegressor(),
        sample_posterior=False,
        max_iter=100,
        missing_values=np.nan,
    )

    res_mice = test_imputer.fit_transform(df_corrupted)
    rmse_mice = utils.mean_absolute_error(
        df_init[mask_mcar], res_mice[mask_mcar], columnwise=False
    )

    test_imputer = models.ImputeRandom()
    res_random = test_imputer.fit_transform(df_corrupted)
    rmse_random = utils.mean_absolute_error(
        df_init[mask_mcar], res_random[mask_mcar], columnwise=False
    )

    assert rmse_mice < rmse_random


def test_impute_MICE_hyperparameters() -> None:
    from sklearn.ensemble import ExtraTreesRegressor

    test_imputer = models.ImputeMICE(
        estimator=ExtraTreesRegressor(),
        sample_posterior=False,
        max_iter=100,
        missing_values=np.nan,
    )

    assert len(test_imputer.get_hyperparams()) == 4
    assert 'estimator' in test_imputer.get_hyperparams()
