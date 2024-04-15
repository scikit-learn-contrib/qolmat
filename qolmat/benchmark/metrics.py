from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy
from sklearn import metrics as skm
import dcor

from qolmat.utils import algebra, utils
from qolmat.utils.exceptions import NotEnoughSamples
from numpy.linalg import LinAlgError

EPS = np.finfo(float).eps

###########################
# Column-wise metrics     #
###########################


def columnwise_metric(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    metric: Callable,
    type_cols: str = "all",
    **kwargs,
) -> pd.Series:
    """For each column, compute a metric score based on the true dataframe
    and the predicted dataframe

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on
    metric : Callable
        metric function
    type_cols : str
        Can be either:
        - `all` to apply the metric to all columns
        - `numerical` to apply the metric to numerical columns only
        - `categorical` to apply the metric to categorical columns only

    Returns
    -------
    pd.Series
        Series of scores for all columns
    """
    try:
        pd.testing.assert_index_equal(df1.columns, df2.columns)
    except AssertionError:
        raise ValueError(
            f"Input dataframes do not have the same columns! ({df1.columns} != {df2.columns})"
        )
    if type_cols == "all":
        cols = df1.columns
    elif type_cols == "numerical":
        cols = utils._get_numerical_features(df1)
    elif type_cols == "categorical":
        cols = utils._get_categorical_features(df1)
    else:
        raise ValueError(f"Value {type_cols} is not valid for parameter `type_cols`!")
    values = {}
    for col in cols:
        df1_col = df1.loc[df_mask[col], col]
        df2_col = df2.loc[df_mask[col], col]
        assert df1_col.notna().all()
        assert df2_col.notna().all()
        values[col] = metric(df1_col, df2_col, **kwargs)

    return pd.Series(values)


def mean_squared_error(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> pd.Series:
    """Mean squared error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
    """
    return columnwise_metric(df1, df2, df_mask, skm.mean_squared_error, type_cols="numerical")


def root_mean_squared_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> pd.Series:
    """Root mean squared error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
    """
    return columnwise_metric(
        df1, df2, df_mask, skm.mean_squared_error, type_cols="numerical", squared=False
    )


def mean_absolute_error(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> pd.Series:
    """Mean absolute error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
    """
    return columnwise_metric(df1, df2, df_mask, skm.mean_absolute_error, type_cols="numerical")


def mean_absolute_percentage_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> pd.Series:
    """Mean absolute percentage error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
    """
    return columnwise_metric(
        df1, df2, df_mask, skm.mean_absolute_percentage_error, type_cols="numerical"
    )


def _weighted_mean_absolute_percentage_error_1D(values1: pd.Series, values2: pd.Series) -> float:
    """Weighted mean absolute percentage error between two series.
    Based on https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Parameters
    ----------
    values1 : pd.Series
        True values
    values2 : pd.Series
        Predicted values

    Returns
    -------
    float
        Weighted mean absolute percentage error
    """
    return (values1 - values2).abs().sum() / values1.abs().sum()


def weighted_mean_absolute_percentage_error(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> pd.Series:
    """Weighted mean absolute percentage error between two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
    """
    return columnwise_metric(
        df1,
        df2,
        df_mask,
        _weighted_mean_absolute_percentage_error_1D,
        type_cols="numerical",
    )


def accuracy(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> pd.Series:
    """
    Matching ratio beetween the two datasets.

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
    """
    return columnwise_metric(
        df1,
        df2,
        df_mask,
        accuracy_1D,
        type_cols="all",
    )


def accuracy_1D(values1: pd.Series, values2: pd.Series) -> float:
    """
    Matching ratio beetween the set of values.

    Parameters
    ----------
    values1 : pd.Series
        True values
    values2 : pd.Series
        Predicted values

    Returns
    -------
    float
        accuracy
    """
    return (values1 == values2).mean()


def dist_wasserstein(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    method: str = "columnwise",
) -> pd.Series:
    """Wasserstein distances between columns of 2 dataframes.
    Wasserstein distance can only be computed columnwise

    Parameters
    ----------
    df1 : pd.DataFrame
        True dataframe
    df2 : pd.DataFrame
        Predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
        wasserstein distances
    """
    if method == "columnwise":
        return columnwise_metric(df1, df2, df_mask, scipy.stats.wasserstein_distance)
    else:
        raise AssertionError(
            f"The parameter of the function wasserstein_distance should be one of"
            f"the following: [`columnwise`], not `{method}`!"
        )


def kolmogorov_smirnov_test_1D(df1: pd.Series, df2: pd.Series) -> float:
    """Compute KS test statistic of the two-sample Kolmogorov-Smirnov test for goodness of fit.
    See more in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html.

    Parameters
    ----------
    df1 : pd.Series
        true series
    df2 : pd.Series
        predicted series

    Returns
    -------
    float
        KS test statistic
    """
    return scipy.stats.ks_2samp(df1, df2)[0]


def kolmogorov_smirnov_test(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> pd.Series:
    """Kolmogorov Smirnov Test for numerical features.
    Lower score means better performance.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
        KS test statistic
    """
    return columnwise_metric(df1, df2, df_mask, kolmogorov_smirnov_test_1D, type_cols="numerical")


def _total_variance_distance_1D(df1: pd.Series, df2: pd.Series) -> float:
    """Compute Total Variance Distance for a categorical feature
    It is based on TVComplement in https://github.com/sdv-dev/SDMetrics

    Parameters
    ----------
    df1 : pd.Series
        true series
    df2 : pd.Series
        predicted series

    Returns
    -------
    float
        Total variance distance
    """
    list_categories = list(set(df1.unique()).union(set(df2.unique())))
    freqs1 = df1.value_counts() / len(df1)
    freqs1 = freqs1.reindex(list_categories, fill_value=0.0)
    freqs2 = df2.value_counts() / len(df2)
    freqs2 = freqs2.reindex(list_categories, fill_value=0.0)
    return (freqs1 - freqs2).abs().sum()


def total_variance_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame
) -> pd.Series:
    """Total variance distance for categorical features
    It is based on TVComplement in https://github.com/sdv-dev/SDMetrics

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
        Total variance distance
    """
    return columnwise_metric(
        df1,
        df2,
        df_mask,
        _total_variance_distance_1D,
        type_cols="categorical",
    )


def _check_same_number_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    if len(df1.columns) != len(df2.columns):
        raise Exception("inputs have to have the same number of columns.")


def _get_correlation_pearson_matrix(df: pd.DataFrame, use_p_value: bool = True) -> pd.DataFrame:
    """Get matrix of correlation values for numerical features
    based on Pearson correlation coefficient or p-value for testing non-correlation.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    use_p_value : bool, optional
        use the p-value instead of the correlation coefficient, by default True

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    cols = df.columns.tolist()
    matrix = np.zeros((len(df.columns), len(df.columns)))
    for idx_1, col_1 in enumerate(cols):
        for idx_2, col_2 in enumerate(cols):
            res = scipy.stats.mstats.pearsonr(df[[col_1]].values, df[[col_2]].values)
            if use_p_value:
                matrix[idx_1, idx_2] = res[1]
            else:
                matrix[idx_1, idx_2] = res[0]

    return pd.DataFrame(matrix, index=cols, columns=cols)


def mean_difference_correlation_matrix_numerical_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    use_p_value: bool = True,
) -> pd.Series:
    """Mean absolute of differences between the correlation matrices of df1 and df2.
    based on Pearson correlation coefficient or p-value for testing non-correlation.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on
    use_p_value : bool, optional
        use the p-value instead of the correlation coefficient, by default True

    Returns
    -------
    pd.Series
        Mean absolute of differences for each feature
    """
    df1 = df1[df_mask].dropna(axis=0)
    df2 = df2[df_mask].dropna(axis=0)

    _check_same_number_columns(df1, df2)

    cols_numerical = utils._get_numerical_features(df1)
    df_corr1 = _get_correlation_pearson_matrix(df1[cols_numerical], use_p_value=use_p_value)
    df_corr2 = _get_correlation_pearson_matrix(df2[cols_numerical], use_p_value=use_p_value)

    diff_corr = (df_corr1 - df_corr2).abs().mean(axis=1)
    return pd.Series(diff_corr, index=cols_numerical)


def _get_correlation_chi2_matrix(data: pd.DataFrame, use_p_value: bool = True) -> pd.DataFrame:
    """Get matrix of correlation values for categorical features
    based on Chi-square test of independence of variables (the test statistic or the p-value).

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    use_p_value : bool, optional
        use the p-value of the test instead of the test statistic, by default True

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    cols = data.columns.tolist()
    matrix = np.zeros((len(data.columns), len(data.columns)))
    for idx_1, col_1 in enumerate(cols):
        for idx_2, col_2 in enumerate(cols):
            freq = data.pivot_table(
                index=col_1, columns=col_2, aggfunc="size", fill_value=0
            ).to_numpy()
            res = scipy.stats.chi2_contingency(freq)
            if use_p_value:
                matrix[idx_1, idx_2] = res[1]
            else:
                matrix[idx_1, idx_2] = res[0]
    return pd.DataFrame(matrix, index=cols, columns=cols)


def mean_difference_correlation_matrix_categorical_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    use_p_value: bool = True,
) -> pd.Series:
    """Mean absolute of differences between the correlation matrix of df1 and df2
    based on Chi-square test of independence of variables (the test statistic or the p-value)

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on
    use_p_value : bool, optional
        use the p-value of the test instead of the test statistic, by default True

    Returns
    -------
    pd.Series
        Mean absolute of differences for each feature
    """
    df1 = df1[df_mask].dropna(axis=0)
    df2 = df2[df_mask].dropna(axis=0)

    _check_same_number_columns(df1, df2)

    cols_categorical = utils._get_categorical_features(df1)
    df_corr1 = _get_correlation_chi2_matrix(df1[cols_categorical], use_p_value=use_p_value)
    df_corr2 = _get_correlation_chi2_matrix(df2[cols_categorical], use_p_value=use_p_value)

    diff_corr = (df_corr1 - df_corr2).abs().mean(axis=1)
    return pd.Series(diff_corr, index=cols_categorical)


def _get_correlation_f_oneway_matrix(
    df: pd.DataFrame,
    cols_categorical: List[str],
    cols_numerical: List[str],
    use_p_value: bool = True,
) -> pd.DataFrame:
    """Get matrix of correlation values between categorical and numerical features
    based on the one-way ANOVA.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    cols_categorical : List[str]
        list categorical columns
    cols_numerical : List[str]
        list numerical columns
    use_p_value : bool, optional
        use the p-value of the test instead of the test statistic, by default True

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    matrix = np.zeros((len(cols_categorical), len(cols_numerical)))
    for idx_cat, col_cat in enumerate(cols_categorical):
        for idx_num, col_num in enumerate(cols_numerical):
            category_group_lists = df.groupby(col_cat)[col_num].apply(list)
            res = scipy.stats.f_oneway(*category_group_lists)
            if use_p_value:
                matrix[idx_cat, idx_num] = res[1]
            else:
                matrix[idx_cat, idx_num] = res[0]
    return pd.DataFrame(matrix, index=cols_categorical, columns=cols_numerical)


def mean_diff_corr_matrix_categorical_vs_numerical_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    use_p_value: bool = True,
) -> pd.Series:
    """Mean absolute of differences between the correlation matrix of df1 and df2
    based on the one-way ANOVA.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on
    use_p_value : bool, optional
        use the p-value of the test instead of the test statistic, by default True

    Returns
    -------
    pd.Series
        Mean absolute of differences for each feature
    """
    df1 = df1[df_mask].dropna(axis=0)
    df2 = df2[df_mask].dropna(axis=0)

    _check_same_number_columns(df1, df2)

    cols_categorical = utils._get_categorical_features(df1)
    cols_numerical = utils._get_numerical_features(df1)
    df_corr1 = _get_correlation_f_oneway_matrix(
        df1, cols_categorical, cols_numerical, use_p_value=use_p_value
    )
    df_corr2 = _get_correlation_f_oneway_matrix(
        df2, cols_categorical, cols_numerical, use_p_value=use_p_value
    )
    diff_corr = (df_corr1 - df_corr2).abs().mean(axis=1)
    return pd.Series(diff_corr, index=cols_categorical)


###########################
# Row-wise metrics        #
###########################


def _sum_manhattan_distances_1D(values: pd.Series) -> float:
    """Sum of Manhattan distances computed for one column
    It is based on https://www.geeksforgeeks.org/sum-manhattan-distances-pairs-points/

    Parameters
    ----------
    values : pd.Series
        Values of a column

    Returns
    -------
    float
        Sum of Manhattan distances
    """
    values = values.sort_values(ascending=True)
    sums_partial = values.shift().fillna(0.0).cumsum()
    differences_partial = values * np.arange(len(values)) - sums_partial
    res = differences_partial.sum()
    return res


def _sum_manhattan_distances(df1: pd.DataFrame) -> float:
    """Sum Manhattan distances between all pairs of rows.
    It is based on https://www.geeksforgeeks.org/sum-manhattan-distances-pairs-points/

    Parameters
    ----------
    df1 : pd.DataFrame

    Returns
    -------
    float
        Sum of Manhattan distances for all pairs of rows.
    """
    cols = df1.columns.tolist()
    result = sum([_sum_manhattan_distances_1D(df1[col]) for col in cols])
    return result


def sum_energy_distances(df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame) -> pd.Series:
    """Sum of energy distances between df1 and df2.
    It is based on https://dcor.readthedocs.io/en/latest/theory.html#

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on

    Returns
    -------
    pd.Series
        Sum of energy distances between df1 and df2.
    """

    # Replace nan in dataframe
    df1 = df1[df_mask].fillna(0.0)
    df2 = df2[df_mask].fillna(0.0)

    # sum of (len_df1 * (len_df1 - 1) / 2) distances for df1
    sum_distances_df1 = _sum_manhattan_distances(df1)
    sum_distances_df2 = _sum_manhattan_distances(df2)

    df = pd.concat([df1, df2])
    sum_distances_df1_df2 = _sum_manhattan_distances(df)
    sum_distance = 2 * sum_distances_df1_df2 - 4 * sum_distances_df1 - 4 * sum_distances_df2

    return pd.Series(sum_distance, index=["All"])


def sum_pairwise_distances(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    metric: str = "cityblock",
) -> float:
    """Sum of pairwise distances based on a predefined metric.
    Metrics are found in this link
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    Parameters
    ----------
    df1 : pd.DataFrame
        First empirical distribution without nans
    df2 : pd.DataFrame
        Second empirical distribution without nans
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on
    metric : str, optional
        distance metric, by default 'cityblock'

    Returns
    -------
    float
        Sum of pairwise distances based on a predefined metric
    """
    df1 = df1[df_mask.any(axis=1)]
    df2 = df2[df_mask.any(axis=1)]
    distances = np.sum(scipy.spatial.distance.cdist(df1, df2, metric=metric))

    return distances


###########################
# Dataframe-wise metrics  #
###########################


def frechet_distance_base(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.Series:
    """Compute the Fréchet distance between two dataframes df1 and df2
    Frechet_distance = || mu_1 - mu_2 ||_2^2 + Tr(Sigma_1 + Sigma_2 - 2(Sigma_1 . Sigma_2)^(1/2))
    It is normalized, df1 and df2 are first scaled by a factor (std(df1) + std(df2)) / 2
    and then centered around (mean(df1) + mean(df2)) / 2
    Based on: Dowson, D. C., and BV666017 Landau. "The Fréchet distance between multivariate normal
    distributions." Journal of multivariate analysis 12.3 (1982): 450-455.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe

    Returns
    -------
    pd.Series
        Frechet distance in a Series object
    """

    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")

    std = (np.std(df1) + np.std(df2) + EPS) / 2
    mu = (np.nanmean(df1, axis=0) + np.nanmean(df2, axis=0)) / 2
    df1 = (df1 - mu) / std
    df2 = (df2 - mu) / std

    means1, cov1 = utils.nan_mean_cov(df1.values)
    means2, cov2 = utils.nan_mean_cov(df2.values)

    distance = algebra.frechet_distance_exact(means1, cov1, means2, cov2)
    return pd.Series(distance, index=["All"])


def frechet_distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    method: str = "single",
    min_n_rows: int = 10,
) -> pd.Series:
    """
    Frechet distance computed using a pattern decomposition. Several variant are implemented:
    - the `single` method relies on a single estimation of the means and covariance matrix. It is
    relevent for MCAR data.
    - the `pattern`method relies on the aggregation of the estimated distance between each
    pattern. It is relevent for MAR data.

    Parameters
    ----------
    df1 : pd.DataFrame
        First empirical ditribution
    df2 : pd.DataFrame
        Second empirical ditribution
    df_mask : pd.DataFrame
        Mask indicating on which values the distance has to computed on
    method: str
        Method used to compute the distance on multivariate datasets with missing values.
        Possible values are `robust` and `pattern`.
    min_n_rows: int
        Minimum number of rows for a KL estimation

    Returns
    -------
    pd.Series
        Series of computed metrics
    """

    if method == "single":
        return frechet_distance_base(df1, df2)
    return pattern_based_weighted_mean_metric(
        df1,
        df2,
        df_mask,
        frechet_distance_base,
        min_n_rows=min_n_rows,
        type_cols="numerical",
    )


def kl_divergence_1D(df1: pd.Series, df2: pd.Series) -> float:
    """Estimation of the Kullback-Leibler divergence between the two 1D empirical distributions
    given by `df1`and `df2`. The samples are binarized using a uniform spacing with 20 bins from
    the smallest to the largest value. Not that this may be a coarse estimation.

    Parameters
    ----------
    df1 : pd.Series
        First empirical distribution
    df2 : pd.Series
        Second empirical distribution

    Returns
    -------
    float
        Kullback-Leibler divergence between the two empirical distributions.
    """
    min_val = min(df1.min(), df2.min())
    max_val = max(df1.max(), df2.max())
    bins = np.linspace(min_val, max_val, 20)
    p = np.histogram(df1, bins=bins, density=True)[0]
    q = np.histogram(df2, bins=bins, density=True)[0]
    return scipy.stats.entropy(p + EPS, q + EPS)


def kl_divergence_gaussian(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Kullback-Leibler divergence estimation based on a Gaussian approximation of both empirical
    distributions

    Parameters
    ----------
    df1 : pd.DataFrame
        First empirical distribution
    df2 : pd.DataFrame
        Second empirical distribution

    Returns
    -------
    pd.Series
        Series of estimated metrics
    """
    cov1 = df1.cov().values
    cov2 = df2.cov().values
    means1 = np.array(df1.mean())
    means2 = np.array(df2.mean())
    try:
        div_kl = algebra.kl_divergence_gaussian_exact(means1, cov1, means2, cov2)
    except LinAlgError:
        raise ValueError(
            "Provided datasets have degenerate colinearities, KL-divergence cannot be computed!"
        )
    return div_kl


def kl_divergence(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    method: str = "columnwise",
    min_n_rows: int = 10,
) -> pd.Series:
    """
    Estimation of the Kullback-Leibler divergence between too empirical distributions. Three
    methods are implemented:
    - columnwise, relying on a uniform binarization and only taking marginals into account
    (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence),
    - gaussian, relying on a Gaussian approximation,

    Parameters
    ----------
    df1 : pd.DataFrame
        First empirical distribution
    df2 : pd.DataFrame
        Second empirical distribution
    df_mask: pd.DataFrame
        Mask indicating on what values the divergence should be computed
    method: str
        Method used to compute the divergence on multivariate datasets with missing values.
        Possible values are `columnwise` and `gaussian`.
    min_n_rows: int
        Minimum number of rows for a KL estimation

    Returns
    -------
    pd.Series
        Kullback-Leibler divergence

    Raises
    ------
    AssertionError
        If the empirical distributions do not have enough samples to estimate a KL divergence.
        Consider using a larger dataset of lowering the parameter `min_n_rows`.
    """
    if method == "columnwise":
        return columnwise_metric(df1, df2, df_mask, kl_divergence_1D, type_cols="numerical")
    elif method == "gaussian":
        return pattern_based_weighted_mean_metric(
            df1,
            df2,
            df_mask,
            kl_divergence_gaussian,
            min_n_rows=min_n_rows,
            type_cols="numerical",
        )
    else:
        raise AssertionError(
            f"The parameter of the function wasserstein_distance should be one of"
            f"the following: [`columnwise`, `gaussian`], not `{method}`!"
        )


def distance_anticorr(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Score based on the distance anticorrelation between two empirical distributions.
    The theoretical basis can be found on dcor documentation:
    https://dcor.readthedocs.io/en/latest/theory.html

    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe representing the first empirical distribution
    df2 : pd.DataFrame
        Dataframe representing the second empirical distribution

    Returns
    -------
    float
        Distance correlation score
    """
    return (1 - dcor.distance_correlation(df1.values, df2.values)) / 2


def distance_anticorr_pattern(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    min_n_rows: int = 10,
) -> pd.Series:
    """Correlation distance computed using a pattern decomposition

    Parameters
    ----------
    df1 : pd.DataFrame
        First empirical ditribution
    df2 : pd.DataFrame
        Second empirical ditribution
    df_mask : pd.DataFrame
        Mask indicating on which values the distance has to computed on
    min_n_rows: int
        Minimum number of rows for a KL estimation

    Returns
    -------
    pd.Series
        Series of computed metrics
    """

    return pattern_based_weighted_mean_metric(
        df1,
        df2,
        df_mask,
        distance_anticorr,
        min_n_rows=min_n_rows,
        type_cols="numerical",
    )


def pattern_based_weighted_mean_metric(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    metric: Callable,
    min_n_rows: int = 10,
    type_cols: str = "all",
    **kwargs,
) -> pd.Series:
    """Compute a mean score based on missing patterns.
    Note that for each pattern, a score is returned by the function metric.
    This code is based on https://www.statsmodels.org/

    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe representing the first empirical distribution, with nans
    df2 : pd.DataFrame
        Dataframe representing the second empirical distribution
    df_mask : pd.DataFrame
        Elements of the dataframes to compute on
    metric : Callable
        metric function
    min_n_rows : int, optional
        minimum number of row allowed for a pattern without nan, by default 10

    Returns
    -------
    pd.Series
        _description_
    """
    if type_cols == "all":
        cols = df1.columns
    elif type_cols == "numerical":
        cols = df1.select_dtypes(include=["number"]).columns
    elif type_cols == "categorical":
        cols = df1.select_dtypes(exclude=["number"]).columns
    else:
        raise ValueError(f"Value {type_cols} is not valid for parameter `type_cols`!")

    if np.any(df_mask & df1.isna()):
        raise ValueError("The argument df1 has missing values on the mask!")
    if np.any(df_mask & df2.isna()):
        raise ValueError("The argument df2 has missing values on the mask!")

    rows_mask = df_mask.any(axis=1)
    scores = []
    weights = []
    df1 = df1[cols].loc[rows_mask]
    df2 = df2[cols].loc[rows_mask]
    df_mask = df_mask[cols].loc[rows_mask]
    max_num_row = 0
    for tup_pattern, df_mask_pattern in df_mask.groupby(df_mask.columns.tolist()):
        ind_pattern = df_mask_pattern.index
        df1_pattern = df1.loc[ind_pattern, list(tup_pattern)]
        max_num_row = max(max_num_row, len(df1_pattern))
        if not any(tup_pattern) or len(df1_pattern) < min_n_rows:
            continue
        df2_pattern = df2.loc[ind_pattern, list(tup_pattern)]
        weights.append(len(df1_pattern) / len(df1))
        scores.append(metric(df1_pattern, df2_pattern, **kwargs))
    if len(scores) == 0:
        raise NotEnoughSamples(max_num_row, min_n_rows)
    return pd.Series(sum([s * w for s, w in zip(scores, weights)]), index=["All"])


def get_metric(name: str) -> Callable:
    dict_metrics: Dict[str, Callable] = {
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "mae": mean_absolute_error,
        "wmape": weighted_mean_absolute_percentage_error,
        "accuracy": accuracy,
        "wasserstein_columnwise": dist_wasserstein,
        "KL_columnwise": partial(kl_divergence, method="columnwise"),
        "KL_gaussian": partial(kl_divergence, method="gaussian"),
        "KS_test": kolmogorov_smirnov_test,
        "correlation_diff": mean_difference_correlation_matrix_numerical_features,
        "energy": sum_energy_distances,
        "frechet": partial(frechet_distance, method="single"),
        "frechet_pattern": partial(frechet_distance, method="pattern"),
        "dist_corr_pattern": distance_anticorr_pattern,
    }
    return dict_metrics[name]
