import abc
import copy
import sys
from typing import Any, Dict, List, Optional, Union

import sklearn.neighbors._base
from sklearn.base import BaseEstimator

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base


import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute._base import _BaseImputer
from statsmodels.tsa import seasonal as tsa_seasonal

from qolmat.benchmark import utils
from qolmat.imputations import em_sampler
from qolmat.imputations.rpca.rpca_noisy import RPCANoisy
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP


class Imputer(_BaseImputer):
    def __init__(
        self,
        groups: List[str] = [],
        columnwise: bool = False,
        shrink: bool = False,
        hyperparams: Dict = {},
    ):
        self.hyperparams_user = hyperparams
        self.hyperparams_optim = {}
        self.hyperparams_local = {}
        self.groups = groups
        self.columnwise = columnwise
        self.shrink = shrink

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform to impute with RPCA methods

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")

        hyperparams = self.hyperparams_user.copy()
        hyperparams.update(self.hyperparams_optim)
        cols_with_nans = df.columns[df.isna().any()]

        if self.groups == []:
            self.ngroups = pd.Series(0, index=df.index).rename("_ngroup")
        else:
            self.ngroups = df.groupby(self.groups).ngroup().rename("_ngroup")

        if self.columnwise:

            # imputed = pd.DataFrame(index=df.index, columns=df.columns)
            df_imputed = df.copy()

            for col in cols_with_nans:
                self.hyperparams_element = {}
                for hyperparam, value in hyperparams.items():
                    if isinstance(value, dict):
                        value = value[col]
                    self.hyperparams_element[hyperparam] = value

                df_imputed[col] = self.impute_element(df[[col]])

        else:
            self.hyperparams_element = hyperparams
            df_imputed = self.impute_element(df)

        if df_imputed.isna().any().any():
            raise AssertionError("Result of imputation contains NaN!")

        return df_imputed

    def fit_transform_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna("median")

    def impute_element(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")
        df = df.copy()
        if self.groups:

            # groupby = utils.custom_groupby(df, self.groups)
            groupby = df.groupby(self.ngroups, group_keys=False)
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


class ImputerMean(Imputer):
    """
    This class implements the implementation by the mean of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeByMean
    >>> imputor = ImputeByMean()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True, shrink=True)
        self.fit_transform_element = pd.DataFrame.mean


class ImputerMedian(Imputer):
    """
    This class implements the implementation by the median of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeByMedian
    >>> imputor = ImputeByMedian()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                         [np.nan, np.nan, np.nan, np.nan],
    >>>                         [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True, shrink=True)
        self.fit_transform_element = pd.DataFrame.median


class ImputerMode(Imputer):
    """
    This class implements the implementation by the mode of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeByMode
    >>> imputor = ImputeByMode()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True, shrink=True)
        self.fit_transform_element = lambda df: df.mode().iloc[0]


class ImputerShuffle(Imputer):
    """
    This class implements the imputation by a random value (from the observed ones) of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeRandom
    >>> imputor = ImputeRandom()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def fit_transform_element(self, df):
        n_missing = df.isna().sum().sum()
        if df.isna().all().all():
            return df
        name = df.columns[0]
        values = df[name]
        values_notna = values.dropna()
        samples = np.random.choice(values_notna, n_missing, replace=True)
        values[values.isna()] = samples
        df_imputed = values.to_frame()
        return df_imputed


class ImputerLOCF(Imputer):
    """
    This class implements a forward imputation, column wise

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeLOCF
    >>> imputor = ImputeLOCF()
    >>> df = pd.DataFrame(data=[[np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5],
    >>>                        [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform by imputing missing values by carrying the last observation forward.
        If the first observation is missing, it is imputed by a NOCB
        """
        df_out = df.copy()
        for col in df:
            df_out[col] = pd.Series.shift(df[col], 1).ffill().bfill()
        return df_out


class ImputerNOCB(Imputer):
    """
    This class implements a backawrd imputation, column wise

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeNOCB
    >>> imputor = ImputeNOCB()
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform by imputing missing values by carrying the next observation backward.
        If the last observation is missing, it is imputed by the median of the series
        """
        df_out = df.copy()
        for col in df:
            df_out[col] = pd.Series.shift(df[col], 1).bfill().ffill()
        return df_out


class ImputerInterpolation(Imputer):
    """
    This class implements a way to impute using some interpolation strategies
    suppoted by pd.Series.interpolate, such as "linear", "slinear", "quadratic", ...
    By default, linear interpolation.
    As for pd.Series.interpolate, if "method" is "spline" or "polynomial",
    an "order" has to be passed.

    Parameters
    ----------
    method : Optional[str] = "linear"
        name of the method for interpolation: "linear", "cubic", "spline", "slinear", ...
        see pd.Series.interpolate for more example.
        By default, the value is set to "linear".
    order : Optional[int]
        order for the spline interpolation

    Examples
    --------
    >>> import numpy as np
    >>> from qolmat.imputations.models import ImputeByInterpolation
    >>> imputor = ImputeByInterpolation(method="spline", order=2)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
        method: str = "linear",
        order: int = None,
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
    This class implements an imputation on residuals.
    The series are de-seasonalised, residuals are imputed, then residuals are re-seasonalised.

    Parameters
    ----------
    period : int
        Period of the series. Must be used if x is not a pandas object or if
        the index of x does not have  a frequency. Overrides default
        periodicity of x if x is a pandas object with a timeseries index.
    model : Optional[str]
        Type of seasonal component "additive" or "multiplicative". Abbreviations are accepted.
        By default, the value is set to "additive"
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends (or the single one
        if two_sided is False) considering this many (+1) closest points.
        If set to 'freq', use `freq` closest points. Setting this parameter
        results in no NaN values in trend or resid components.
    method_interpolation : str
        methof for the residuals interpolation

    Examples
    --------
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
        groups: List[str] = [],
        period: int = None,
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
        """
        Fit/transform missing values on residuals.
        """
        if len(df.columns) != 1:
            raise AssertionError(
                "Function ImputerResiduals.fit_transform_element expects a dataframe df with one column"
            )
        name = df.columns[0]
        df = df[name]
        if df.isna().all():
            return np.nan
        result = tsa_seasonal.seasonal_decompose(
            df.interpolate().bfill().ffill(),
            model=self.model_tsa,
            period=self.period,
            extrapolate_trend=self.extrapolate_trend,
        )

        residuals = result.resid

        residuals[df.isna()] = np.nan
        residuals = residuals.interpolate(method=self.method_interpolation).ffill().bfill()
        df_result = pd.DataFrame({name: result.seasonal + result.trend + residuals})

        return df_result


class ImputerKNN(Imputer):
    """
    This class implements an imputation by the k-nearest neighbors, column wise

    Parameters
    ----------
    k : int
        number of nearest neighbors

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeKNN
    >>> imputor = ImputeKNN(k=2)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
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

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform by imputing missing values with the KNN method.

        Parameters
        ----------
        signal : pd.DataFrame
            DataFrame to impute

        Returns
        -------
        pd.DataFrame
            imputed DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")

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
    It uses sklearn.impute.IterativeImputer, see the docs for more information about the arguments.

    Parameters
    ----------
    estimator : Optional[] = LinearRegression()
        estimator for imputing a column based on the other
    sample_posterior : Optional[bool] = False
        By default, the value is set to False
    max_iter : Optional[int] = 100
        By default, the value is set to 100
    missing_values : Optional[float] = np.nan
        By default, the value is set to np.nan

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeMICE
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> imputor = ImputeMICE(estimator=ExtraTreesRegressor(),
    >>>                           sample_posterior=False,
    >>>                           max_iter=100, missing_values=np.nan)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        **hyperparams,
    ) -> None:
        super().__init__(groups=groups, columnwise=False, hyperparams=hyperparams)
        self.estimator = estimator

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform using an iterative imputer and a specific estimator

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")

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
    It imputes each Series with missing value within a DataFrame using the complete ones.

    Parameters
    ----------
    model :
        regression model

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeRegressor
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> imputor = ImputeRegressor(model=ExtraTreesRegressor())
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                       [np.nan, np.nan, 2, 3],
    >>>                       [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                       columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(df)
    """

    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        fit_on_nan: bool = False,
        **hyperparams,
    ):
        super().__init__(groups=groups, hyperparams=hyperparams)
        self.columnwise = False
        self.estimator = estimator
        self.fit_on_nan = fit_on_nan

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

        df_imputed = df.copy()

        cols_with_nans = df.columns[df.isna().any()]
        cols_without_nans = df.columns[df.notna().all()]

        for col in cols_with_nans:
            hyperparams = {}
            for hyperparam, value in self.hyperparams_element.items():
                if isinstance(value, dict):
                    value = value[col]
                hyperparams[hyperparam] = value

            # model = copy.deepcopy(self.estimator)
            # for hyperparam, value in hyperparams.items():
            #     setattr(model, hyperparam, value)

            if self.fit_on_nan:
                X = df.drop(columns=col)
            else:
                X = df[cols_without_nans].drop(columns=col, errors="ignore")
            y = df[col]
            is_na = y.isna()
            if X.empty:
                y_imputed = pd.Series(y.mean(), index=y.index)
            else:
                self.estimator.fit(X[~is_na], y[~is_na])
                y_imputed = self.estimator.predict(X[is_na])
            df_imputed.loc[is_na, col] = y_imputed

        return df_imputed


class ImputerStochasticRegressor(Imputer):
    """
    This class implements a stochastic regression imputer in the multivariate case.
    It imputes each Series with missing value within a DataFrame using the complete ones.

    Parameters
    ----------
    model :
        regression model

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeStochasticRegressor
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> imputer = ImputeStochasticRegressor(estimator=ExtraTreesRegressor)
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 2, 3],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
    """

    def __init__(
        self, groups: List[str] = [], estimator: Optional[BaseEstimator] = None, **hyperparams
    ) -> None:
        super().__init__(groups=groups, hyperparams=hyperparams)
        self.estimator = estimator

    def fit_transform_element(self, df: pd.DataFrame) -> pd.Series:
        """
        Fit/transform using a (specified) regression model + stochastic

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """
        df_imp = df.copy()
        cols_with_nans = df.columns[df.isna().any()]
        cols_without_nans = df.columns[df.notna().all()]

        if len(cols_without_nans) == 0:
            raise Exception("There must be at least one column without missing values.")

        for col in cols_with_nans:
            X = df[cols_without_nans]
            y = df[col]
            is_na = y.isna()
            self.estimator.fit(X[~is_na], y[~is_na])
            y_pred = self.estimator.predict(X)
            std_error = (y_pred[~is_na] - y[~is_na]).std()
            random_pred = np.random.normal(size=len(y), loc=y_pred, scale=std_error)
            df_imp.loc[is_na, col] = random_pred[is_na]

        return df_imp


class ImputerRPCA(Imputer):
    """
    This class implements the RPCA imputation

    Parameters
    ----------
    method : str
        name of the RPCA method:
            "PCP" for basic RPCA, bad at imputing
            "noisy" for noisy RPCA, with possible regularisations
    columnwise : bool
        for RPCA method to be applied columnwise (with reshaping of each column into an array)
        or to be applied directly on the dataframe. By default, the value is set to False.
    """

    def __init__(
        self,
        groups: List[str] = [],
        method: str = "noisy",
        columnwise: bool = False,
        **hyperparams,
    ) -> None:
        super().__init__(groups=groups, columnwise=columnwise, hyperparams=hyperparams)

        self.method = method

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform to impute with RPCA methods

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")

        if self.method == "PCP":
            model = RPCAPCP(**self.hyperparams_element)
        elif self.method == "noisy":
            model = RPCANoisy(**self.hyperparams_element)
        else:
            raise ValueError("Argument method must be `PCP` or `noisy`!")

        X_imputed = model.fit_transform(df.values)
        df_imputed = pd.DataFrame(X_imputed, index=df.index, columns=df.columns)

        return df_imputed


class ImputerEM(Imputer):
    def __init__(
        self,
        groups: List[str] = [],
        method: Optional[str] = "multinormal",
        columnwise: bool = False,
        **hyperparams,
    ):
        super().__init__(groups=groups, columnwise=columnwise, hyperparams=hyperparams)
        self.method = method

    def fit_transform_element(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method == "multinormal":
            model = em_sampler.MultiNormalEM(**self.hyperparams_element)
        elif self.method == "VAR1":
            model = em_sampler.VAR1EM(**self.hyperparams_element)
        else:
            raise ValueError("Strategy '{strategy}' is not handled by ImputeEM!")
        X = df.values
        model.fit(X)

        X_transformed = model.transform(X)
        df_transformed = pd.DataFrame(X_transformed, columns=df.columns, index=df.index)
        return df_transformed

    # def fit(self, df):
    #     X = df.values
    #     self.model.fit(X)
    #     return self

    # def transform(self, df):
    #     X = df.values
    #     X_transformed = self.model.transform(X)
    #     df_transformed = pd.DataFrame(X_transformed, columns=df.columns, index=df.index)
    #     return df_transformed
