import warnings
from typing import Dict, List, Optional, Union
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn import utils as sku
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute._base import _BaseImputer
from statsmodels.tsa import seasonal as tsa_seasonal

from qolmat.imputations import em_sampler
from qolmat.imputations.rpca.rpca_noisy import RPCANoisy
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP


class Imputer(_BaseImputer):
    """Base class for all imputers.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    columnwise : bool, optional
        If True, the imputer will be computed for each column, else it will be computed on the
        whole dataframe, by default False
    shrink : bool, optional
        Indicates if the elementwise imputation method returns a single value, by default False
    hyperparams : Dict, optional
        Hyperparameters to be passed to the imputer, for example in the case when the imputer
        requires a regression model.
        If a dictionary of values is provided, each value is a global hyperparameter.
        If a nested dictionary of dictionaries is provided and `columnwise` is True, it should be
        indexed by the dataset column names.
        This allows to provide different hyperparameters for each column.
        By default {}
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    """

    def __init__(
        self,
        columnwise: bool = False,
        shrink: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        missing_values=np.nan,
        groups: List[str] = [],
        hyperparams: Dict = {},
    ):
        self.columnwise = columnwise
        self.shrink = shrink
        self.random_state = random_state
        self.missing_values = missing_values
        self.groups = groups
        self.hyperparams = hyperparams

    def _more_tags(self):
        """Define tags for scikit-learn"""

        return {
            "allow_nan": True,
            "requires_fit": False,
            "_xfail_checks": {
                "check_parameters_default_constructible": "The imputer need Dict as a parammeter",
                "check_no_attributes_set_in_init": """The imputer can define an attribute
                modifiable in init""",
            },
        }

    def fit(self, X, y: pd.DataFrame = None):
        X = self._validate_data(X, force_all_finite="allow-nan")
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Returns a dataframe with same shape as `df`, unchanged values, where all nans are replaced
        by non-nan values.
        Depending on the imputer parameters, the dataframe can be imputed with columnwise and/or
        groupwise methods.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to impute.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.
        """
        self.fit(X)

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input has to be a pandas.DataFrame or numpy.ndarray.")
        df = pd.DataFrame(X)
        for column in df:
            if df[column].isnull().all():
                raise ValueError("Input contains a column full of NaN")
        self.rng = sku.check_random_state(self.random_state)
        if hasattr(self, "estimator") and hasattr(self.estimator, "random_state"):
            self.estimator.random_state = self.rng

        hyperparams = self.hyperparams.copy()
        if hasattr(self, "hyperparams_optim"):
            hyperparams.update(self.hyperparams_optim)
        cols_with_nans = df.columns[df.isna().any()]

        if self.groups == []:
            self.ngroups_ = pd.Series(0, index=df.index).rename("_ngroup")
        else:
            self.ngroups_ = df.groupby(self.groups).ngroup().rename("_ngroup")

        if self.columnwise:
            df_imputed = df.copy()

            for col in cols_with_nans:
                self.hyperparams_element = {}
                for hyperparam, value in hyperparams.items():
                    if isinstance(value, dict):
                        value = value[col]
                    self.hyperparams_element[hyperparam] = value

                df_imputed[col] = self.impute_element(df[[col]])

        else:
            if any(isinstance(value, dict) for value in hyperparams.values()):
                raise AssertionError("hyperparams contains a dictionary. Columnwise must be True.")
            self.hyperparams_element = hyperparams
            df_imputed = self.impute_element(df)

        if df_imputed.isna().any().any():
            raise AssertionError("Result of imputation contains NaN!")

        return df_imputed

    def fit_transform_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute `df` by the median of each column if it still contains missing values.
        This can introduce data leakage if unchecked.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with missing values.

        Returns
        -------
        pd.DataFrame
            Dataframe df imputed by the median of each column.
        """
        return df.fillna(df.median())

    def impute_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute `df` by applying the specialized method `fit_transform_element` on each group, if
        groups have been given. If the method leaves nan, `fit_transform_fallback` is called in
        order to return a dataframe without nan.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute

        Returns
        -------
        pd.DataFrame
            Imputed dataframe or column

        Raises
        ------
        ValueError
            Input has to be a pandas.DataFrame.
        """
        # Impute `df` by applying the specialized method `fit_transform_element` on each group, if
        # groups have been given.
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")
        df = df.copy()
        if self.groups:
            # groupby = utils.custom_groupby(df, groups)
            groupby = df.groupby(self.ngroups_, group_keys=False)
            if self.shrink:
                imputation_values = groupby.transform(self.fit_transform_element)
            else:
                imputation_values = groupby.apply(self.fit_transform_element)
        else:
            imputation_values = self.fit_transform_element(df)

        df = df.fillna(imputation_values)
        # fill na by applying imputation method without groups
        if df.isna().any().any():
            imputation_values = self.fit_transform_fallback(df)
            df = df.fillna(imputation_values)

        return df

    @abstractmethod
    def fit_transform_element(self, df: pd.DataFrame):
        return df


class ImputerOracle(Imputer):
    """
    Perfect imputer, requires to know real values.

    Used as a reference to evaluate imputation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing real values.
    groups : List[str], optional
        List of column names to group by, by default []
    """

    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.df = df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Impute df with corresponding known values

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute
        Returns
        -------
        pd.DataFrame
            dataframe imputed with premasked values
        """
        self.fit(X)
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input has to be a pandas.DataFrame or numpy.ndarray.")
        df = pd.DataFrame(X)
        return df.fillna(self.df)


class ImputerMean(Imputer):
    """Impute by the mean of the column.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerMean()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                         [np.nan, np.nan, np.nan, np.nan],
    ...                         [1, 2, 2, 5],
    ...                         [2, 2, 2, 2]],
    ...                         columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
           var1      var2      var3      var4
    0  1.000000  1.000000  1.000000  1.000000
    1  1.333333  1.666667  1.666667  2.666667
    2  1.000000  2.000000  2.000000  5.000000
    3  2.000000  2.000000  2.000000  2.000000
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True, shrink=True)

    def _more_tags(self):
        return {"allow_nan": True, "requires_fit": False}

    def fit_transform_element(self, df: pd.DataFrame):
        return pd.DataFrame.mean(df)


class ImputerMedian(Imputer):
    """Impute by the median of the column.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerMedian()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                         [np.nan, np.nan, np.nan, np.nan],
    ...                         [1, 2, 2, 5],
    ...                         [2, 2, 2, 2]],
    ...                         columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   1.0   2.0   2.0   2.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True, shrink=True)

    def fit_transform_element(self, df: pd.DataFrame):
        return pd.DataFrame.median(df)


class ImputerMode(Imputer):
    """Impute by the mode of the column, which is the most represented value.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerMode()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                         [np.nan, np.nan, np.nan, np.nan],
    ...                         [1, 2, 2, 5],
    ...                         [2, 2, 2, 2]],
    ...                         columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   1.0   2.0   2.0   1.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True, shrink=True)

    def fit_transform_element(self, df: pd.DataFrame):
        return df.mode().iloc[0]


class ImputerShuffle(Imputer):
    """Impute using random samples from the considered column.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    random_state : Union[None, int, np.random.RandomState], optional
        Determine the randomness of the imputer, by default None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerShuffle(random_state=42)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                         [np.nan, np.nan, np.nan, np.nan],
    ...                         [1, 2, 2, 5],
    ...                         [2, 2, 2, 2]],
    ...                         columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   2.0   1.0   2.0   2.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        groups: List[str] = [],
        random_state: Union[None, int, np.random.RandomState] = None,
    ) -> None:
        super().__init__(groups=groups, columnwise=True, random_state=random_state)

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        n_missing = df.isna().sum().sum()
        if df.isna().all().all():
            return df
        name = df.columns[0]
        values = df[name]
        values_notna = values.dropna()
        samples = self.rng.choice(values_notna, n_missing, replace=True)
        values[values.isna()] = samples
        df_imputed = values.to_frame()
        return df_imputed


class ImputerLOCF(Imputer):
    """Impute by the last available value of the column. Relevent for time series.

    If the first observations are missing, it is imputed by a NOCB

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerLOCF()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                         [np.nan, np.nan, np.nan, np.nan],
    ...                         [1, 2, 2, 5],
    ...                         [2, 2, 2, 2]],
    ...                         columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   1.0   1.0   1.0   1.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in df:
            df_out[col] = df[col].ffill().bfill()
        return df_out


class ImputerNOCB(Imputer):
    """Impute by the next available value of the column. Relevent for time series.
    If the last observation is missing, it is imputed by a LOCF.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerNOCB()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                         [np.nan, np.nan, np.nan, np.nan],
    ...                         [1, 2, 2, 5],
    ...                         [2, 2, 2, 2]],
    ...                         columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   1.0   2.0   2.0   5.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in df:
            df_out[col] = df[col].bfill().ffill()
        return df_out


class ImputerInterpolation(Imputer):
    """
    This class implements a way to impute time series using some interpolation strategies
    suppoted by pd.Series.interpolate, such as "linear", "slinear", "quadratic", ...
    By default, linear interpolation.
    As for pd.Series.interpolate, if "method" is "spline" or "polynomial",
    an "order" has to be passed.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    method : Optional[str] = "linear"
        name of the method for interpolation: "linear", "cubic", "spline", "slinear", ...
        see pd.Series.interpolate for more example.
        By default, the value is set to "linear".
    order : Optional[int]
        order for the spline interpolation
    col_time : Optional[str]
        Name of the column representing the time index to use for the interpolation. If None, the
        index is used assuming it is one-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerInterpolation(method="spline", order=2)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                        [np.nan, np.nan, np.nan, np.nan],
    ...                        [1, 2, 2, 5],
    ...                        [2, 2, 2, 2]],
    ...                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
           var1      var2      var3      var4
    0  1.000000  1.000000  1.000000  1.000000
    1  0.666667  1.666667  1.666667  4.666667
    2  1.000000  2.000000  2.000000  5.000000
    3  2.000000  2.000000  2.000000  2.000000
    """

    def __init__(
        self,
        groups: List[str] = [],
        method: str = "linear",
        order: Optional[int] = None,
        col_time: Optional[str] = None,
    ) -> None:
        super().__init__(groups=groups, columnwise=True)
        self.method = method
        self.order = order
        self.col_time = col_time

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        index = df.index
        if self.col_time is None:
            df = df.reset_index(drop=True)
        else:
            df.index = df.index.get_level_values(self.col_time)
        df_imputed = df.interpolate(method=self.method, order=self.order)
        df_imputed = df_imputed.ffill().bfill()
        df_imputed.index = index
        return df_imputed


class ImputerResiduals(Imputer):
    """
    This class implements an imputation method based on a STL decomposition.
    The series are de-seasonalised, de-trended, residuals are imputed, then residuals are
    re-seasonalised and re-trended.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    period : int
        Period of the series. Must be used if x is not a pandas object or if
        the index of x does not have  a frequency. Overrides default
        periodicity of x if x is a pandas object with a timeseries index.
    model_tsa : Optional[str]
        Type of seasonal component "additive" or "multiplicative". Abbreviations are accepted.
        By default, the value is set to "additive"
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends (or the single one
        if two_sided is False) considering this many (+1) closest points.
        If set to 'freq', use `freq` closest points. Setting this parameter
        results in no NaN values in trend or resid components.
    method_interpolation : str
        method for the residuals interpolation

    Examples
    --------
    TODO review/remake this exemple
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeOnResiduals
    >>> df = pd.DataFrame(index=pd.date_range('2015-01-01','2020-01-01'))
    >>> mean = 5
    >>> offset = 10
    >>> df['y'] = np.cos(df.index.dayofyear/365*2*np.pi - np.pi)*mean + offset
    >>> trend = 5
    >>> df['y'] = df['y'] + trend*np.arange(0,df.shape[0])/df.shape[0]
    >>> noise_mean = 0
    >>> noise_var = 2
    >>> df['y'] = df['y'] + np.random.normal(noise_mean, noise_var, df.shape[0])
    >>> np.random.seed(100)
    >>> mask = np.random.choice([True, False], size=df.shape)
    >>> df = df.mask(mask)
    >>> imputor = ImputeOnResiduals(period=365, model="additive")
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        period: int,
        groups: List[str] = [],
        model_tsa: Optional[str] = "additive",
        extrapolate_trend: Optional[Union[int, str]] = "freq",
        method_interpolation: Optional[str] = "linear",
    ):
        super().__init__(groups=groups, columnwise=True)
        self.model_tsa = model_tsa
        self.period = period
        self.extrapolate_trend = extrapolate_trend
        self.method_interpolation = method_interpolation

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        name = df.columns[0]
        values = df[name]
        if values.isna().all():
            return np.nan
        result = tsa_seasonal.seasonal_decompose(
            # df.interpolate().bfill().ffill(),
            values.fillna(0),
            model=self.model_tsa,
            period=self.period,
            extrapolate_trend=self.extrapolate_trend,
        )

        residuals = result.resid

        residuals[values.isna()] = np.nan
        residuals = residuals.interpolate(method=self.method_interpolation).ffill().bfill()
        df_result = pd.DataFrame({name: result.seasonal + result.trend + residuals})
        return df_result


class ImputerKNN(Imputer):
    """
    This class implements an imputation by the k-nearest neighbors.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerKNN(n_neighbors=2)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                        [np.nan, np.nan, np.nan, np.nan],
    ...                        [1, 2, 2, 5],
    ...                        [2, 2, 2, 2]],
    ...                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
           var1      var2      var3      var4
    0  1.000000  1.000000  1.000000  1.000000
    1  1.333333  1.666667  1.666667  2.666667
    2  1.000000  2.000000  2.000000  5.000000
    3  2.000000  2.000000  2.000000  2.000000
    """

    def __init__(
        self,
        groups: List[str] = [],
        n_neighbors: int = 5,
        weights: str = "distance",
        **hyperparams,
    ) -> None:
        super().__init__(groups=groups, columnwise=False, hyperparams=hyperparams)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.hyperparams_optim: Dict = {}

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="nan_euclidean",
        )
        results = imputer.fit_transform(df)
        return pd.DataFrame(data=results, columns=df.columns, index=df.index)


class ImputerMICE(Imputer):
    """
    This class implements an iterative imputer in the multivariate case.
    It imputes each Series within a DataFrame multiple times using an iteration of fits
    and transformations to reach a stable state of imputation each time.
    It uses sklearn.impute.IterativeImputer, see the docs for more information about the
    arguments.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    estimator : Optional[] = LinearRegression()
        Estimator for imputing a column based on the others
    random_state : Union[None, int, np.random.RandomState], optional
        Determine the randomness of the imputer, by default None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> imputer = imputers.ImputerMICE(estimator=ExtraTreesRegressor(),
    ...                                random_state=42,
    ...                                sample_posterior=False,
    ...                                max_iter=100, missing_values=np.nan)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                        [np.nan, np.nan, np.nan, np.nan],
    ...                        [1, 2, 2, 5],
    ...                        [2, 2, 2, 2]],
    ...                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   1.0   2.0   2.0   5.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        random_state: Union[None, int, np.random.RandomState] = None,
        **hyperparams,
    ) -> None:
        super().__init__(
            groups=groups,
            columnwise=False,
            hyperparams=hyperparams,
            random_state=random_state,
        )
        self.estimator = estimator
        self.hyperparams_optim: Dict = {}

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        iterative_imputer = IterativeImputer(estimator=self.estimator, **self.hyperparams_element)
        res = iterative_imputer.fit_transform(df.values)
        imputed = pd.DataFrame(columns=df.columns)
        for ind, col in enumerate(imputed.columns):
            imputed[col] = res[:, ind]
        imputed.index = df.index
        return imputed


class ImputerRegressor(Imputer):
    """
    This class implements a regression imputer in the multivariate case.
    It imputes each column using a single fit-predict for a given estimator, based on the colunms
    which have no missing values.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    estimator : BaseEstimator, optional
        Estimator for imputing a column based on the others
    fit_on_nan : bool, optional
        TODO : merge with GSA

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> imputer = imputers.ImputerRegressor(model=ExtraTreesRegressor())
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                        [np.nan, np.nan, np.nan, np.nan],
    ...                        [1, 2, 2, 5],
    ...                        [2, 2, 2, 2]],
    ...                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
           var1      var2      var3      var4
    0  1.000000  1.000000  1.000000  1.000000
    1  1.333333  1.666667  1.666667  2.666667
    2  1.000000  2.000000  2.000000  5.000000
    3  2.000000  2.000000  2.000000  2.000000
    """

    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "column",
        random_state: Union[None, int, np.random.RandomState] = None,
        **hyperparams,
    ):
        super().__init__(groups=groups, hyperparams=hyperparams, random_state=random_state)
        self.columnwise = False
        self.estimator = estimator
        self.handler_nan = handler_nan
        self.hyperparams_optim: Dict = {}

    def get_params_fit(self) -> Dict:
        return {}

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform using a (specified) regression model

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """

        df_imputed = df.apply(pd.DataFrame.median, result_type="broadcast", axis=0)
        cols_with_nans = df.columns[df.isna().any()]

        for col in cols_with_nans:
            hyperparams = {}
            for hyperparam, value in self.hyperparams_element.items():
                if isinstance(value, dict):
                    value = value[col]
                hyperparams[hyperparam] = value

            # Define the Train and Test set
            X = df.drop(columns=col, errors="ignore")
            y = df[col]

            # Selects only the valid values in the Train Set according to the chosen method
            is_valid = pd.Series(True, index=df.index)
            if self.handler_nan == "fit":
                pass
            elif self.handler_nan == "row":
                is_valid = ~X.isna().any(axis=1)
            elif self.handler_nan == "column":
                X = X.dropna(how="any", axis=1)
            else:
                raise ValueError(
                    f"Value '{self.handler_nan}' is not correct for argument `handler_nan'"
                )

            # Selects only non-NaN values for the Test Set
            is_na = y.isna()

            # Train the model according to an ML or DL method and after predict the imputation
            if X.empty:
                y_imputed = pd.Series(y.mean(), index=y.index)
            else:
                hp = self.get_params_fit()
                self.estimator.fit(X[(~is_na) & is_valid], y[(~is_na) & is_valid], **hp)
                y_imputed = self.estimator.predict(X[is_na & is_valid])
                y_imputed = pd.Series(y_imputed.flatten())

            # Adds the imputed values
            df_imputed.loc[~is_na, col] = y[~is_na]
            # if isinstance(y_imputed, pd.Series):
            #     y_reshaped = y_imputed
            # else:
            #     y_reshaped = y_imputed.flatten()
            df_imputed.loc[is_na & is_valid, col] = y_imputed

        return df_imputed


class ImputerRPCA(Imputer):
    """
    This class implements the Robust Principal Component Analysis imputation.

    The imputation minimizes a loss function combining a low-rank criterium on the dataframe and
    a L1 penalization on the residuals.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    method : str
        Name of the RPCA method:
            "PCP" for basic RPCA, bad at imputing
            "noisy" for noisy RPCA, with possible regularisations, wihch is recommended since
            it is more stable
    columnwise : bool
        For the RPCA method to be applied columnwise (with reshaping of
        each column into an array)
        or to be applied directly on the dataframe. By default, the value is set to False.
    """

    def __init__(
        self,
        groups: List[str] = [],
        method: str = "noisy",
        columnwise: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        **hyperparams,
    ) -> None:
        super().__init__(
            groups=groups,
            columnwise=columnwise,
            hyperparams=hyperparams,
            random_state=random_state,
        )

        self.method = method
        self.hyperparams_optim: Dict = {}

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")

        if self.method == "PCP":
            model = RPCAPCP(**self.hyperparams_element)
        elif self.method == "noisy":
            model = RPCANoisy(**self.hyperparams_element)
        else:
            raise ValueError("Argument method must be `PCP` or `noisy`!")

        X = df.values.T
        M, A = model.decompose_rpca_signal(X)
        df_imputed = pd.DataFrame((M + A).T, index=df.index, columns=df.columns)
        df_imputed = df.where(~df.isna(), df_imputed)

        return df_imputed


class ImputerEM(Imputer):
    """
    This class implements an imputation method based on joint modelling and an inference using a
    Expectation-Minimization algorithm.

    Parameters
    ----------
    groups : List[str], optional
        List of column names to group by, by default []
    method : {'multinormal', 'VAR1'}, default='multinormal'
        Method defining the hypothesis made on the data distribution. Possible values:
        - 'multinormal' : the data points a independent and uniformly distributed following a
        multinormal distribution
        - 'VAR1' : the data is a time series modeled by a VAR(1) process
    columnwise : bool
        If False, correlations between variables will be used, which is advised.
        If True, each column is imputed independently. For the multinormal case each
        value will be imputed by the mean up to a noise with fixed noise, for the VAR1 case the
        imputation will be a noisy temporal interpolation.
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    """

    def __init__(
        self,
        groups: List[str] = [],
        model: Optional[str] = "multinormal",
        columnwise: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        **hyperparams,
    ):
        super().__init__(
            groups=groups,
            columnwise=columnwise,
            hyperparams=hyperparams,
            random_state=random_state,
        )
        self.model = model
        self.hyperparams_optim: Dict = {}

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model == "multinormal":
            model = em_sampler.MultiNormalEM(random_state=self.rng, **self.hyperparams_element)
        elif self.model == "VAR1":
            model = em_sampler.VAR1EM(random_state=self.rng, **self.hyperparams_element)
        else:
            raise ValueError(f"Model '{self.model}' is not handled by ImputeEM!")
        X = df.values.T
        model.fit(X)

        X_transformed = model.transform(X)
        df_transformed = pd.DataFrame(X_transformed, columns=df.columns, index=df.index)
        return df_transformed
