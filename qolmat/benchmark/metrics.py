from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn import metrics as skm
from sklearn.ensemble import BaseEnsemble
from sklearn.preprocessing import StandardScaler

EPS = np.finfo(float).eps

###########################
# Column-wise metrics     #
###########################


def columnwise_metric(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame, metric: Callable, **kwargs
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

    Returns
    -------
    pd.Series
        Series of scores for all columns
    """
    values = {}
    for col in df1.columns:
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
    return columnwise_metric(df1, df2, df_mask, skm.mean_squared_error)


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
    return columnwise_metric(df1, df2, df_mask, skm.mean_squared_error, squared=False)


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
    return columnwise_metric(df1, df2, df_mask, skm.mean_absolute_error)


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
    return columnwise_metric(df1, df2, df_mask, skm.mean_absolute_percentage_error)


def wasserstein_distance(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame, method: str = "columnwise"
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


def density_from_rf(
    df: pd.DataFrame, estimator: BaseEnsemble, df_est: Optional[pd.DataFrame] = None
):
    """Estimates the density of the empirical distribution given by df at the sample points given
    by df_est. The estimation uses an random forest estimator and relies on the average number of
    samples in the leaf corresponding to each estimation point.

    Parameters
    ----------
    df : pd.DataFrame
        Empirical distribution which density should be estimated
    estimator : BaseEnsemble
        Estimator defining the forest upon which is based the density counting.
    df_est : pd.DataFrame, optional
        Sample points of the estimation, by default None
        If None, the density is estimated at the points given by `df`.

    Returns
    -------
    pd.Series
        Series of floats providing the normalized density
    """
    if df_est is None:
        df_est = df.copy()
    counts = pd.Series(0, index=df_est.index)
    df_leafs = pd.DataFrame(estimator.apply(df))
    df_leafs_est = pd.DataFrame(estimator.apply(df_est))
    for i_tree in range(estimator.n_estimators):
        leafs = df_leafs[i_tree].rename("id_leaf")
        leafs_est = df_leafs_est[i_tree].rename("id_leaf")
        counts_leafs = leafs.value_counts().rename("count")
        df_merge = pd.merge(leafs_est.reset_index(), counts_leafs.reset_index(), on="id_leaf")
        df_merge = df_merge.set_index("index")
        counts += df_merge["count"]
    counts /= counts.sum()
    return counts


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


def kl_divergence(
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame, method: str = "columnwise"
) -> pd.Series:
    """
    Estimation of the Kullback-Leibler divergence between too empirical distributions. Three
    methods are implemented:
    - columnwise, relying on a uniform binarization and only taking marginals into account
    (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence),
    - gaussian, relying on a Gaussian approximation,
    - random_forest, experimental

    Parameters
    ----------
    df1 : pd.DataFrame
        First empirical distribution
    df2 : pd.DataFrame
        Second empirical distribution
    df_mask: pd.DataFrame
        Mask indicating on what values the divergence should be computed
    method:

    Returns
    -------
    Kullback-Leibler divergence : Union[float, pd.Series]
    """
    if method == "columnwise":
        return columnwise_metric(df1, df2, df_mask, kl_divergence_1D)
    elif method == "gaussian":
        n_variables = len(df1.columns)
        cov1 = df1.cov()
        cov2 = df2.cov()
        mean1 = df1.mean()
        mean2 = df2.mean()
        L1, lower1 = scipy.linalg.cho_factor(cov1)
        L2, lower2 = scipy.linalg.cho_factor(cov2)
        M = scipy.linalg.solve(L2, L1)
        y = scipy.linalg.solve(L2, mean2 - mean1)
        norm_M = (M**2).sum().sum()
        norm_y = (y**2).sum()
        term_diag_L = 2 * np.sum(np.log(np.diagonal(L2) / np.diagonal(L1)))
        print(norm_M, "-", n_variables, "+", norm_y, "+", term_diag_L)
        div_kl = 0.5 * (norm_M - n_variables + norm_y + term_diag_L)
        return pd.Series(div_kl, index=df1.columns)
    elif method == "random_forest":
        # df_1 = StandardScaler().fit_transform(df1[df_mask.any(axis=1)])
        # df_2 = StandardScaler().fit_transform(df2[df_mask.any(axis=1)])
        n_estimators = 1000
        # estimator = sklearn.ensemble.RandomForestClassifier(
        #     n_estimators=n_estimators, max_depth=10
        # )
        # X = pd.concat([df1, df2])
        # y = pd.concat([pd.Series([False] * len(df1)), pd.Series([True] * len(df2))])
        # estimator.fit(X, y)
        estimator = sklearn.ensemble.RandomTreesEmbedding(n_estimators=n_estimators, max_depth=8)
        estimator.fit(df1)
        counts1 = density_from_rf(df1, estimator, df_est=df2)
        counts2 = density_from_rf(df2, estimator, df_est=df2)
        div_kl = np.mean(np.log(counts1 / counts2) * counts1 / counts2)
        return pd.Series(div_kl, index=df1.columns)
    else:
        raise AssertionError(
            f"The parameter of the function wasserstein_distance should be one of"
            f"the following: [`columnwise`, `gaussian`], not `{method}`!"
        )


def _get_numerical_features(df1: pd.DataFrame) -> List[str]:
    """Get numerical features from dataframe

    Parameters
    ----------
    df1 : pd.DataFrame

    Returns
    -------
    List[str]
        List of numerical features

    Raises
    ------
    Exception
        No numerical feature is found
    """
    cols_numerical = df1.select_dtypes(include=np.number).columns.tolist()
    if len(cols_numerical) == 0:
        raise Exception("No numerical feature is found.")
    else:
        return cols_numerical


def _get_categorical_features(df1: pd.DataFrame) -> List[str]:
    """Get categorical features from dataframe

    Parameters
    ----------
    df1 : pd.DataFrame

    Returns
    -------
    List[str]
        List of categorical features

    Raises
    ------
    Exception
        No categorical feature is found
    """

    cols_numerical = df1.select_dtypes(include=np.number).columns.tolist()
    cols_categorical = [col for col in df1.columns.to_list() if col not in cols_numerical]
    if len(cols_categorical) == 0:
        raise Exception("No categorical feature is found.")
    else:
        return cols_categorical


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
    cols_numerical = _get_numerical_features(df1)
    return columnwise_metric(
        df1[cols_numerical],
        df2[cols_numerical],
        df_mask[cols_numerical],
        kolmogorov_smirnov_test_1D,
    )


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
    cols_categorical = _get_categorical_features(df1)
    return columnwise_metric(
        df1[cols_categorical],
        df2[cols_categorical],
        df_mask[cols_categorical],
        _total_variance_distance_1D,
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
            res = scipy.stats.mstats.pearsonr(
                df[col_1].array.reshape(-1, 1), df[col_2].array.reshape(-1, 1)
            )
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

    cols_numerical = _get_numerical_features(df1)
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

    cols_categorical = _get_categorical_features(df1)
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

    cols_categorical = _get_categorical_features(df1)
    cols_numerical = _get_numerical_features(df1)
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
    df1: pd.DataFrame, df2: pd.DataFrame, df_mask: pd.DataFrame, metric: str = "cityblock"
) -> pd.Series:
    """Sum of pairwise distances based on a predefined metric.
    Metrics are found in this link
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    metric : str, optional
        distance metric, by default 'cityblock'

    Returns
    -------
    pd.Series
        Sum of pairwise distances based on a predefined metric
    """
    distances = np.sum(
        scipy.spatial.distance.cdist(
            df1[df_mask].fillna(0.0), df2[df_mask].fillna(0.0), metric=metric
        )
    )

    return pd.Series(distances, index=["All"])


###########################
# Dataframe-wise metrics  #
###########################


def frechet_distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df_mask: pd.DataFrame,
    normalized: Optional[bool] = False,
) -> pd.Series:
    """Compute the Fréchet distance between two dataframes df1 and df2
    frechet_distance = || mu_1 - mu_2 ||_2^2 + Tr(Sigma_1 + Sigma_2 - 2(Sigma_1 . Sigma_2)^(1/2))
    if normalized, df1 and df_ are first scaled by a factor
        (std(df1) + std(df2)) / 2
    and then centered around
        (mean(df1) + mean(df2)) / 2

    Dowson, D. C., and BV666017 Landau. "The Fréchet distance between multivariate normal
    distributions."
    Journal of multivariate analysis 12.3 (1982): 450-455.

    Parameters
    ----------
    df1 : pd.DataFrame
        true dataframe
    df2 : pd.DataFrame
        predicted dataframe
    normalized: Optional[bool]
        if the data has to be normalised. By default, is set to False

    Returns
    -------
    frechet_distance : float
    """

    if df1.shape != df2.shape:
        raise Exception("inputs have to be of same dimensions.")

    df_true = df1[df_mask.any(axis=1)]
    df_pred = df2[df_mask.any(axis=1)]

    if normalized:
        std = (np.std(df_true) + np.std(df_pred) + EPS) / 2
        mu = (np.nanmean(df_true, axis=0) + np.nanmean(df_pred, axis=0)) / 2
        df_true = (df_true - mu) / std
        df_pred = (df_pred - mu) / std

    mu_true = np.nanmean(df_true, axis=0)
    sigma_true = np.ma.cov(np.ma.masked_invalid(df_true), rowvar=False).data
    mu_pred = np.nanmean(df_pred, axis=0)
    sigma_pred = np.ma.cov(np.ma.masked_invalid(df_pred), rowvar=False).data

    ssdiff = np.sum((mu_true - mu_pred) ** 2.0)
    product = np.array(sigma_true @ sigma_pred)
    if product.ndim < 2:
        product = product.reshape(-1, 1)
    covmean = scipy.linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    frechet_dist = ssdiff + np.trace(sigma_true + sigma_pred - 2.0 * covmean)

    if normalized:
        return pd.Series((frechet_dist / df_true.shape[0]), index=["All"])
    else:
        return pd.Series(np.repeat(frechet_dist, len(df1.columns)))
