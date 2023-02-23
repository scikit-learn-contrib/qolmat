import sys
import warnings
from typing import Dict, List, Optional, Union

import sklearn.neighbors._base
from sklearn.experimental import enable_iterative_imputer

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base

from functools import partial

import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute._base import _BaseImputer
from statsmodels.tsa.seasonal import seasonal_decompose

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
import missingpy

from qolmat.benchmark import utils
from qolmat.imputations import em_sampler
from qolmat.imputations.rpca.pcp_rpca import RPCA
from qolmat.imputations.rpca.temporal_rpca import OnlineTemporalRPCA, TemporalRPCA


class ImputeColumnWise(_BaseImputer):
    """
    This class implements the imputation column wise.

    Parameters
    ----------
    groups : Optional[List]
        names for the custom groupby
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        self.groups = groups

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform the Imputer to the dataset by fitting with specified values

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

        df_imputed = df.copy()
        cols_with_nans = df_imputed.columns[df_imputed.isna().any()]
        for col in cols_with_nans:
            if self.groups:
                groupby = utils.custom_groupby(df, self.groups)
                imputation_values = groupby[col].transform(self.apply_imputation)
            else:
                imputation_values = self.apply_imputation(df[col])

            df_imputed[col] = df_imputed[col].fillna(imputation_values)

            # fill na by applying imputation method without groups
            if df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(self.apply_imputation(df_imputed[col]))

        if df_imputed.isna().any().any():
            print("Number of null by col")
            print(df_imputed.isna().sum())
            warnings.warn("Problem: there are still nan in the columns to be imputed")
        return df_imputed

    def get_hyperparams(self) -> Dict[str, Union[str, float, int]]:
        """
        Return the hyperparameters, if relevant

        Returns
        -------
        Dict[str, Union[str, float, int]]
            dictionary with hyperparameters and their value
        """
        return self.__dict__


class ImputeByMean(ImputeColumnWise):
    """
    This class implements the implementation by the mean of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeByMean
    >>> imputor = ImputeByMean()
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(groups=groups)
        self.apply_imputation = np.nanmean


class ImputeByMedian(ImputeColumnWise):
    """
    This class implements the implementation by the median of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeByMedian
    >>> imputor = ImputeByMedian()
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                         [np.nan, np.nan, np.nan, np.nan],
    >>>                         [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(groups=groups)
        self.apply_imputation = np.nanmedian

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform the Imputer to the dataset by fitting with the median of each column

        Parameters
        ----------
        signal : pd.Series
            series to impute

        Returns
        -------
        pd.Series
            imputed series
        """
        col = signal.name
        signal = signal.reset_index()
        imputed = signal[col].fillna(
            utils.custom_groupby(signal, self.groups)[col].apply("median")
        )
        return imputed


class ImputeByMode(ImputeColumnWise):
    """
    This class implements the implementation by the mode of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeByMode
    >>> imputor = ImputeByMode()
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(groups=groups)

        def get_mode(x):
            if x.isna().all():
                return np.nan
            return x.mode()[0]

        self.apply_imputation = get_mode


class ImputeRandom(ImputeColumnWise):
    """
    This class implements the imputation by a random value (from th eobserved ones) of each column

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeRandom
    >>> imputor = ImputeRandom()
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(groups=groups)

        def get_random(x: pd.Series) -> pd.Series:
            n_missing = x.isna().sum()
            if x.notna().sum() == 0:
                return x
            samples = np.random.choice(x[x.notna()], n_missing, replace=True)
            imputed = x.copy()
            imputed[imputed.isna()] = samples
            return imputed

        self.apply_imputation = get_random


class ImputeLOCF(ImputeColumnWise):
    """
    This class implements a forward imputation, column wise

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeLOCF
    >>> imputor = ImputeLOCF()
    >>> X = pd.DataFrame(data=[[np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5],
    >>>                        [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(groups=groups)

        def get_LOCF(x: pd.Series):
            """
            Fit/transform by imputing missing values by carrying the last observation forward.
            If the first observation is missing, it is imputed by the median of the series
            """
            x_out = pd.Series.shift(x, 1).ffill().bfill()
            return x_out

        self.apply_imputation = get_LOCF


class ImputeNOCB(ImputeColumnWise):
    """
    This class implements a backawrd imputation, column wise

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeNOCB
    >>> imputor = ImputeNOCB()
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        groups: Optional[List[str]] = [],
    ) -> None:
        super().__init__(groups=groups)

        def get_NOCB(x: pd.Series):
            """
            Fit/transform by imputing missing values by carrying the next observation backward.
            If the last observation is missing, it is imputed by the median of the series
            """
            x_out = pd.Series.shift(x, -1).ffill().bfill()
            return x_out

        self.apply_imputation = get_NOCB


class ImputeByInterpolation(ImputeColumnWise):
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
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self, groups: Optional[List[str]] = [], method: str = "linear", order: int = None
    ) -> None:
        super().__init__(groups=groups)
        self.method = method
        self.order = order
        self.apply_imputation = lambda x: x.interpolate(method=method, order=order).ffill().bfill()


class ImputeOnResiduals(ImputeColumnWise):
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
        groups: Optional[List[str]] = [],
        period: int = None,
        model: Optional[str] = "additive",
        extrapolate_trend: Optional[Union[int, str]] = "freq",
        method_interpolation: Optional[str] = "linear",
    ):
        super().__init__(groups=groups)
        self.model = model
        self.period = period
        self.extrapolate_trend = extrapolate_trend
        self.method_interpolation = method_interpolation

        def get_resid(x, model, period, extrapolate_trend, method_interpolation):
            """
            Fit/transform missing values on residuals.
            """
            if x.isna().all():
                return np.nan
            result = seasonal_decompose(
                x.interpolate().bfill().ffill(),
                model=model,
                period=period,
                extrapolate_trend=extrapolate_trend,
            )

            residuals = result.resid
            residuals[x.isnull()] = np.nan
            residuals = residuals.interpolate(method=method_interpolation).ffill().bfill()
            return result.seasonal + result.trend + residuals

        self.apply_imputation = partial(
            get_resid,
            model=self.model,
            period=self.period,
            extrapolate_trend=self.extrapolate_trend,
            method_interpolation=self.method_interpolation,
        )


class ImputeKNN(_BaseImputer):
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
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        **kwargs,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
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
            n_neighbors=self.n_neighbors, weights=self.weights, metric="nan_euclidean"
        )
        results = imputer.fit_transform(df)
        return pd.DataFrame(data=results, columns=df.columns, index=df.index)

    def get_hyperparams(self) -> Dict[str, int]:
        """
        Get the value of the hyperparameter of the method, i.e. number of nearest neighbors

        Returns
        -------
        Dict[str, int]
            number of nearest neighbors
        """
        return {"k": self.k}


class ImputeRPCA(_BaseImputer):
    """
    This class implements the RPCA imputation

    Parameters
    ----------
    method : str
        name of the RPCA method:
            "PCP" for basic RPCA
            "temporal" for temporal RPCA, with regularisations
            "online" for online RPCA
    multivariate : bool
        for RPCA method to be applied collumn wise (with resizing of matrix)
        or to be applied directly on the dataframe. By default, the value is set to False
    TO DO
    """

    def __init__(self, method: str = "temporal", multivariate: bool = False, **kwargs) -> None:
        self.multivariate = multivariate
        self.method = method

        if method == "PCP":
            self.rpca = RPCA()
        elif method == "temporal":
            self.rpca = TemporalRPCA()
        elif method == "onlinetemporal":
            self.rpca = OnlineTemporalRPCA()
        for name, value in kwargs.items():
            setattr(self.rpca, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit/transform to impute wiht RPCA methods

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to impute

        Returns
        -------
        pd.DataFrame
            imputed dataframe
        """

        if self.multivariate:
            imputed, _, _ = self.rpca.fit_transform(signal=df.values)
            imputed = pd.DataFrame(imputed, columns=df.columns)
        else:
            imputed = pd.DataFrame()
            cols_with_nans = df.columns[df.isna().any()]
            for col in cols_with_nans:
                print(col)
                imputed_signal, _, _ = self.rpca.fit_transform(signal=df[col].values)
                imputed[col] = imputed_signal
        imputed.index = df.index

        return imputed

    def get_hyperparams(self) -> Dict[str, Union[str, float, int]]:
        """
        Get the hyperparameters of the RPCA imputer

        Returns
        -------
        Dict[str, Union[str, float, int]]
            dictonary with the hyperparameters and their value
        """
        return {
            **{"method": self.method, "multivariate": self.multivariate},
            **self.rpca.__dict__,
        }


class ImputeEM(_BaseImputer):
    def __init__(
        self,
        strategy: Optional[str] = "mle",
        method: Optional[str] = "multinormal",
        max_iter_em: Optional[int] = 200,
        n_iter_ou: Optional[int] = 50,
        ampli: Optional[int] = 1,
        random_state: Optional[int] = 123,
        dt: Optional[float] = 2e-2,
        tolerance: Optional[float] = 1e-4,
        stagnation_threshold: Optional[float] = 5e-3,
        stagnation_loglik: Optional[float] = 2,
    ):
        if method == "multinormal":
            self.model = em_sampler.ImputeMultiNormalEM(
                strategy=strategy,
                max_iter_em=max_iter_em,
                n_iter_ou=n_iter_ou,
                ampli=ampli,
                random_state=random_state,
                dt=dt,
                tolerance=tolerance,
                stagnation_threshold=stagnation_threshold,
                stagnation_loglik=stagnation_loglik,
            )
        elif method == "VAR1":
            self.model = em_sampler.ImputeVAR1EM(
                strategy=strategy,
                max_iter_em=max_iter_em,
                n_iter_ou=n_iter_ou,
                ampli=ampli,
                random_state=random_state,
                dt=dt,
                tolerance=tolerance,
                stagnation_threshold=stagnation_threshold,
                stagnation_loglik=stagnation_loglik,
            )
        else:
            raise ValueError("Strategy '{strategy}' is not handled by ImputeEM!")

    def fit(self, df):
        X = df.values
        self.model.fit(X)
        return self

    def transform(self, df):
        X = df.values
        X_transformed = self.model.transform(X)
        df_transformed = pd.DataFrame(X_transformed, columns=df.columns, index=df.index)
        return df_transformed


class ImputeMICE(_BaseImputer):
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
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, np.nan, np.nan],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                         columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

        iterative_imputer = IterativeImputer(**(self.kwargs))
        res = iterative_imputer.fit_transform(df.values)
        imputed = pd.DataFrame(columns=df.columns)
        for ind, col in enumerate(imputed.columns):
            imputed[col] = res[:, ind]
        imputed.index = df.index
        return imputed

    def get_hyperparams(self):
        return self.__dict__["kwargs"]


class ImputeRegressor(_BaseImputer):
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
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                       [np.nan, np.nan, 2, 3],
    >>>                       [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                       columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model

        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
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
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input has to be a pandas.DataFrame.")

        df_imputed = df.copy()

        cols_with_nans = df.columns[df.isna().any()]
        cols_without_nans = df.columns[df.notna().all()]

        if len(cols_without_nans) == 0:
            raise Exception("There must be at least one column without missing values.")

        for col in cols_with_nans:
            X = df[cols_without_nans]
            y = df[col]
            is_na = y.isna()
            self.model.fit(X[~is_na], y[~is_na])
            df_imputed.loc[is_na, col] = self.model.predict(X[is_na])

        return df_imputed

    def get_hyperparams(self):
        """
        Get the hyperparameters of the RPCA imputer

        Returns
        -------
        Dict[str, Union[str, float, int]]
            dictonary with the hyperparameters and their value
        """
        return self.__dict__

    def create_features(self, df):
        return None


class ImputeStochasticRegressor(_BaseImputer):
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
    >>> imputor = ImputeStochasticRegressor(model=ExtraTreesRegressor())
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 2, 3],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(self, estimator, **kwargs) -> None:
        self.estimator = estimator

        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
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

    def get_hyperparams(self):
        """
        Get the hyperparameters of the RPCA imputer

        Returns
        -------
        Dict[str, Union[str, float, int]]
            dictonary with the hyperparameters and their value
        """
        return self.__dict__

    def create_features(self, df):
        return None


class ImputeMissForest(_BaseImputer):
    """
    This class implements an imputation for multivariate data with MissForest

    Parameters
    ----------
    max_features: int, optional (default = 10)
        The maximum iterations of the imputation process. Each column with a
        missing value is imputed exactly once in a given iteration.
    n_estimators : integer, optional (default=100)
        The number of trees in the forest.
    criterion: str
        The function to measure the quality of a split.The first element of
        the tuple is for the Random Forest Regressor (for imputing numerical
        variables) while the second element is for the Random Forest
        Classifier (for imputing categorical variables).
    missing_values : np.nan, integer, optional (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
        `int(max_features * n_features)` features are considered at each
        split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.


    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeMissForest
    >>> imputor = ImputeMissForest()
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 2, 3],
    >>>                        [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        criterion: Optional[str] = "squared_error",
        n_estimators: Optional[int] = 100,
        missing_values: Optional[Union[int, str]] = np.nan,
        max_features: Optional[Union[int, float, str]] = 1.0,
        verbose: Optional[int] = 0,
    ) -> None:
        self.max_features = max_features
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.missing_values = missing_values
        self.verbose = verbose

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        imputer = missingpy.MissForest(
            max_features=self.max_features,
            criterion=self.criterion,
            n_estimators=self.n_estimators,
            missing_values=self.missing_values,
            verbose=0,
        )

        if isinstance(df, np.ndarray):
            return imputer.fit_transform(df)
        elif isinstance(df, pd.DataFrame):
            imputed = imputer.fit_transform(df.values)
            return pd.DataFrame(data=imputed, columns=df.columns, index=df.index)
        else:
            raise ValueError("Input array is not a list, np.array, nor pd.DataFrame.")
