import copy
from functools import partial
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Self
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
import pandas as pd
import sklearn as skl
from sklearn import utils as sku
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute._base import _BaseImputer
from sklearn.utils.validation import (
    _check_feature_names_in,
    _num_samples,
    check_array,
    check_is_fitted,
)
from statsmodels.tsa import seasonal as tsa_seasonal

from qolmat.imputations import em_sampler
from qolmat.imputations.rpca import rpca, rpca_noisy, rpca_pcp
from qolmat.imputations import softimpute
from qolmat.utils import utils
from qolmat.utils.exceptions import NotDataFrame, TypeNotHandled
from qolmat.utils.utils import HyperValue


class _Imputer(_BaseImputer):
    """
    Base class for all imputers.

    Parameters
    ----------
    columnwise : bool, optional
        If True, the imputer will be computed for each column, else it will be computed on the
        whole dataframe, by default False
    shrink : bool, optional
        Indicates if the elementwise imputation method returns a single value, by default False
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    imputer_params: Tuple[str, ...]
        List of parameters of the imputer, which can be specified globally or columnwise
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    """

    def __init__(
        self,
        columnwise: bool = False,
        shrink: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        imputer_params: Tuple[str, ...] = (),
        groups: Tuple[str, ...] = (),
    ):
        self.columnwise = columnwise
        self.shrink = shrink
        self.random_state = random_state
        self.imputer_params = imputer_params
        self.groups = groups
        self.missing_values = np.nan

    def get_hyperparams(self, col: Optional[str] = None):
        """
        Filter hyperparameters based on the specified column, the dictionary keys in the form
        name_params/column are only relevent for the specified column and are filtered accordingly.

        Parameters
        ----------
        col : str
            The column name to filter hyperparameters.

        Returns
        -------
        dict
            A dictionary containing filtered hyperparameters.

        """
        hyperparams = {}
        for key in self.imputer_params:
            value = getattr(self, key)
            if "/" not in key:
                name_param = key
                if name_param not in hyperparams:
                    hyperparams[name_param] = value
            elif col is not None:
                name_param, col2 = key.split("/")
                if col2 == col:
                    hyperparams[name_param] = value
        return hyperparams

    def _check_dataframe(self, X: NDArray):
        """
        Checks that the input X is a dataframe, otherwise raises an error.

        Parameters
        ----------
        X : NDArray
            Array-like to process

        Raises
        ------
        ValueError
            Input has to be a pandas.DataFrame.
        """
        if not isinstance(X, (pd.DataFrame)):
            raise NotDataFrame(type(X))

    def _more_tags(self):
        """
        This method indicates that this class allows inputs with categorical data and nans. It
        modifies the behaviour of the functions checking data.
        """
        return {"X_types": ["2darray", "categorical", "string"], "allow_nan": True}

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        """
        Fit the imputer on X.

        Parameters
        ----------
        X : pd.DataFrame
            Data matrix on which the Imputer must be fitted.

        Returns
        -------
        self : Self
            Returns self.
        """

        df = utils._validate_input(X)
        self.n_features_in_ = len(df.columns)

        for column in df:
            if df[column].isnull().all():
                raise ValueError("Input contains a column full of NaN")

        self.columns_ = tuple(df.columns)
        self._rng = sku.check_random_state(self.random_state)
        if hasattr(self, "estimator") and hasattr(self.estimator, "random_state"):
            self.estimator.random_state = self._rng

        if self.groups:
            self.ngroups_ = df.groupby(list(self.groups)).ngroup().rename("_ngroup")
        else:
            self.ngroups_ = pd.Series(0, index=df.index).rename("_ngroup")

        self._setup_fit()
        if self.columnwise:
            for col in df.columns:
                self._fit_allgroups(df[[col]], col=col)
        else:
            self._fit_allgroups(df)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with same shape as `X`, unchanged values, where all nans are replaced
        by non-nan values. Depending on the imputer parameters, the dataframe can be imputed with
        columnwise and/or groupwise methods.
        Also works for numpy arrays, returning numpy arrays, but the use of pandas dataframe is
        advised.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to impute.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.
        """

        df = utils._validate_input(X)
        if tuple(df.columns) != self.columns_:
            raise ValueError(
                """The number of features is different from the counterpart in fit.
                Reshape your data"""
            )

        for column in df:
            if df[column].isnull().all():
                raise ValueError("Input contains a column full of NaN")

        cols_with_nans = df.columns[df.isna().any()]

        if cols_with_nans.empty:
            df_imputed = df
        else:
            if self.columnwise:
                df_imputed = df.copy()
                for col in cols_with_nans:
                    df_imputed[col] = self._transform_allgroups(df[[col]], col=col)
            else:
                df_imputed = self._transform_allgroups(df)

        if isinstance(X, (np.ndarray)):
            df_imputed = df_imputed.to_numpy()

        return df_imputed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Returns a dataframe with same shape as `X`, unchanged values, where all nans are replaced
        by non-nan values.
        Depending on the imputer parameters, the dataframe can be imputed with columnwise and/or
        groupwise methods.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to impute.

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.
        """
        self.fit(X)
        return self.transform(X)

    def _fit_transform_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute `df` by the median of each column if it still contains missing values.
        This can introduce data leakage for forward imputers if unchecked.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with missing values.

        Returns
        -------
        pd.DataFrame
            Dataframe df imputed by the median of each column.
        """
        self._check_dataframe(df)
        cols_with_nan = df.columns[df.isna().any()]
        for col in cols_with_nan:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def _fit_allgroups(self, df: pd.DataFrame, col: str = "__all__") -> Self:
        """
        Fits the Imputer either on a column, for a columnwise setting, on or all columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"

        Returns
        -------
        Self
            Returns self.

        Raises
        ------
        ValueError
            Input has to be a pandas.DataFrame.
        """

        self._check_dataframe(df)
        fun_on_col = partial(self._fit_element, col=col)
        if self.groups:
            groupby = df.groupby(self.ngroups_, group_keys=False)
            self._dict_fitting[col] = groupby.apply(fun_on_col).to_dict()
        else:
            self._dict_fitting[col] = {0: fun_on_col(df)}

        return self

    def _setup_fit(self) -> None:
        """
        Setup step of the fit function, before looping over the columns.
        """
        self._dict_fitting: Dict[str, Any] = dict()
        return

    def _apply_groupwise(self, fun: Callable, df: pd.DataFrame, **kwargs) -> Any:
        """
        Applies the function `fun`in a groupwise manner to the dataframe `df`.


        Parameters
        ----------
        fun : Callable
            Function applied groupwise to the dataframe with arguments kwargs
        df : pd.DataFrame
            Dataframe on which the function is applied

        Returns
        -------
        Any
            Depends on the function signature
        """
        self._check_dataframe(df)
        fun_on_col = partial(fun, **kwargs)
        if self.groups:
            groupby = df.groupby(self.ngroups_, group_keys=False)
            if self.shrink:
                return groupby.transform(fun_on_col)
            else:
                return groupby.apply(fun_on_col)
        else:
            return fun_on_col(df)

    def _transform_allgroups(self, df: pd.DataFrame, col: str = "__all__") -> pd.DataFrame:
        """
        Impute `df` by applying the specialized method `transform_element` on each group, if
        groups have been given. If the method leaves nan, `fit_transform_fallback` is called in
        order to return a dataframe without nan.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"

        Returns
        -------
        pd.DataFrame
            Imputed dataframe or column

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        df = df.copy()
        imputation_values = self._apply_groupwise(self._transform_element, df, col=col)

        df = df.fillna(imputation_values)
        # fill na by applying imputation method without groups
        if df.isna().any().any():
            imputation_values = self._fit_transform_fallback(df)
            df = df.fillna(imputation_values)

        return df

    @abstractmethod
    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0) -> Any:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return self.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        return self

    @abstractmethod
    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        return df


class ImputerOracle(_Imputer):
    """
    Perfect imputer, requires to know real values.

    Used as a reference to evaluate imputation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing real values.
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def set_solution(self, df: pd.DataFrame):
        """Sets the true values to be returned by the oracle.

        Parameters
        ----------
        X : pd.DataFrame
            True dataset with mask
        """
        self.df_solution = df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        df = utils._validate_input(X)

        if tuple(df.columns) != self.columns_:
            raise ValueError(
                """The number of features is different from the counterpart in fit.
                Reshape your data"""
            )
        if hasattr(self, "df_solution"):
            df_imputed = df.fillna(self.df_solution)
        else:
            warnings.warn("OracleImputer not initialized! Returning imputation with zeros")
            df_imputed = df.fillna(0)

        if isinstance(X, (np.ndarray)):
            df_imputed = df_imputed.to_numpy()
        return df_imputed


class ImputerSimple(_Imputer):
    """
    Impute each column by its mean, its median or its mode (if its categorical).

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> imputer = imputers.ImputerSimple()
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

    def __init__(self, groups: Tuple[str, ...] = (), strategy="median") -> None:
        super().__init__(groups=groups, columnwise=True, shrink=False)
        self.strategy = strategy

    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0) -> Any:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted KNN model

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        if pd.api.types.is_numeric_dtype(df[col]):
            model = skl.impute.SimpleImputer(strategy=self.strategy)
        else:
            model = skl.impute.SimpleImputer(strategy="most_frequent")
        return model.fit(df[[col]])

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending on self.groups
        and self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        model = self._dict_fitting[col][ngroup]
        X_imputed = model.fit_transform(df)
        return pd.DataFrame(data=X_imputed, columns=df.columns, index=df.index)


class ImputerShuffle(_Imputer):
    """
    Impute using random samples from the considered column.

    Parameters
    ----------
    groups: Tuple[str, ...]
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
        groups: Tuple[str, ...] = (),
        random_state: Union[None, int, np.random.RandomState] = None,
    ) -> None:
        super().__init__(groups=groups, columnwise=True, random_state=random_state)

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        n_missing = df.isna().sum().sum()
        if df.isna().all().all():
            return df
        name = df.columns[0]
        values = df[name]
        values_notna = values.dropna()
        samples = self._rng.choice(values_notna, n_missing, replace=True)
        values[values.isna()] = samples
        df_imputed = values.to_frame()
        return df_imputed


class ImputerLOCF(_Imputer):
    """
    Impute by the last available value of the column. Relevent for time series.

    If the first observations are missing, it is imputed by a NOCB

    Parameters
    ----------
    groups: Tuple[str, ...]
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
        groups: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        df_out = df.copy()
        for col in df:
            df_out[col] = df[col].ffill().bfill()
        return df_out


class ImputerNOCB(_Imputer):
    """
    Impute by the next available value of the column. Relevent for time series.
    If the last observation is missing, it is imputed by a LOCF.

    Parameters
    ----------
    groups: Tuple[str, ...]
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
        groups: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(groups=groups, columnwise=True)

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        df_out = df.copy()
        for col in df:
            df_out[col] = df[col].bfill().ffill()
        return df_out


class ImputerInterpolation(_Imputer):
    """
    This class implements a way to impute time series using some interpolation strategies
    suppoted by pd.Series.interpolate, such as "linear", "slinear", "quadratic", ...
    By default, linear interpolation.
    As for pd.Series.interpolate, if "method" is "spline" or "polynomial",
    an "order" has to be passed.

    Parameters
    ----------
    groups: Tuple[str, ...]
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
        groups: Tuple[str, ...] = (),
        method: str = "linear",
        order: Optional[int] = None,
        col_time: Optional[str] = None,
    ) -> None:
        super().__init__(imputer_params=("method", "order"), groups=groups, columnwise=True)
        self.method = method
        self.order = order
        self.col_time = col_time

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams(col=col)
        index = df.index
        if self.col_time is None:
            df = df.reset_index(drop=True)
        else:
            df.index = df.index.get_level_values(self.col_time)
        df_imputed = df.interpolate(**hyperparams)
        df_imputed = df_imputed.ffill().bfill()
        df_imputed.index = index
        return df_imputed


class ImputerResiduals(_Imputer):
    """
    This class implements an imputation method based on a STL decomposition.
    The series are de-seasonalised, de-trended, residuals are imputed, then residuals are
    re-seasonalised and re-trended.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    period : int
        Period of the series. Must be used if x is not a pandas object or if
        the index of x does not have a frequency. Overrides default
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
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.imputers import ImputerResiduals
    >>> np.random.seed(100)
    >>> df = pd.DataFrame(index=pd.date_range('2015-01-01','2020-01-01'))
    >>> mean = 5
    >>> offset = 10
    >>> df['y'] = np.cos(df.index.dayofyear/365*2*np.pi - np.pi)*mean + offset
    >>> trend = 5
    >>> df['y'] = df['y'] + trend*np.arange(0,df.shape[0])/df.shape[0]
    >>> noise_mean = 0
    >>> noise_var = 2
    >>> df['y'] = df['y'] + np.random.normal(noise_mean, noise_var, df.shape[0])
    >>> mask = np.random.choice([True, False], size=df.shape)
    >>> df = df.mask(mask)
    >>> imputor = ImputerResiduals(period=365, model_tsa="additive")
    >>> imputor.fit_transform(df)
                        y
    2015-01-01   1.501210
    2015-01-02   5.691061
    2015-01-03   4.404106
    2015-01-04   3.531540
    2015-01-05   3.129532
    ...               ...
    2019-12-28  10.288054
    2019-12-29  10.632659
    2019-12-30  14.900671
    2019-12-31  12.957837
    2020-01-01  12.780517
    <BLANKLINE>
    [1827 rows x 1 columns]
    """

    def __init__(
        self,
        period: int = 1,
        groups: Tuple[str, ...] = (),
        model_tsa: Optional[str] = "additive",
        extrapolate_trend: Optional[Union[int, str]] = "freq",
        method_interpolation: Optional[str] = "linear",
    ):
        super().__init__(
            imputer_params=(
                "model_tsa",
                "period",
                "extrapolate_trend",
                "method_interpolation",
            ),
            groups=groups,
            columnwise=True,
        )
        self.model_tsa = model_tsa
        self.period = period
        self.extrapolate_trend = extrapolate_trend
        self.method_interpolation = method_interpolation

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams(col=col)
        name = df.columns[0]
        values = df[df.columns[0]]
        values_interp = (
            values.interpolate(method=hyperparams["method_interpolation"]).ffill().bfill()
        )
        result = tsa_seasonal.seasonal_decompose(
            values_interp,
            model=hyperparams["model_tsa"],
            period=hyperparams["period"],
            extrapolate_trend=hyperparams["extrapolate_trend"],
        )

        residuals = result.resid

        residuals[values.isna()] = np.nan
        residuals = (
            residuals.interpolate(method=hyperparams["method_interpolation"]).ffill().bfill()
        )
        df_result = pd.DataFrame({name: result.seasonal + result.trend + residuals})
        return df_result


class ImputerKNN(_Imputer):
    """
    This class implements an imputation by the k-nearest neighbors.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {`uniform`, `distance`}, callable or None, default=`uniform`
        Weight function used in prediction.  Possible values:
            - `uniform` : uniform weights. All points in each neighborhood
                are weighted equally.
            - `distance` : weight points by the inverse of their distance.
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
        groups: Tuple[str, ...] = (),
        n_neighbors: int = 5,
        weights: str = "distance",
    ) -> None:
        super().__init__(
            imputer_params=("n_neighbors", "weights"), groups=groups, columnwise=False
        )
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0) -> KNNImputer:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted KNN model

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        assert col == "__all__"
        hyperparameters = self.get_hyperparams()
        model = KNNImputer(metric="nan_euclidean", **hyperparameters)
        model = model.fit(df)
        return model

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        assert col == "__all__"
        model = self._dict_fitting["__all__"][ngroup]
        X_imputed = model.fit_transform(df)
        return pd.DataFrame(data=X_imputed, columns=df.columns, index=df.index)


class ImputerMICE(_Imputer):
    """
    Wrapper of the class sklearn.impute.IterativeImputer in our framework. This imputer relies
    on a estimator which is iteratively

    Parameters
    ----------
    groups : Tuple[str, ...], optional
        _description_, by default ()
    estimator : Optional[BaseEstimator], optional
        _description_, by default None
    random_state : Union[None, int, np.random.RandomState], optional
        _description_, by default None
    sample_posterior : bool, optional
        _description_, by default False
    max_iter : int, optional
        _description_, by default 100
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        estimator: Optional[BaseEstimator] = None,
        random_state: Union[None, int, np.random.RandomState] = None,
        sample_posterior=False,
        max_iter=100,
    ) -> None:
        super().__init__(
            imputer_params=("sample_posterior", "max_iter"),
            groups=groups,
            columnwise=False,
            random_state=random_state,
        )
        self.estimator = estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter

    def _fit_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> IterativeImputer:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted KNN model

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        assert col == "__all__"
        hyperparameters = self.get_hyperparams()
        model = IterativeImputer(estimator=self.estimator, **hyperparameters)
        model = model.fit(df)
        self.n_iter_ = model.n_iter_
        return model

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending on self.groups
        and self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """

        self._check_dataframe(df)
        assert col == "__all__"
        model = self._dict_fitting["__all__"][ngroup]
        X_imputed = model.fit_transform(df)
        return pd.DataFrame(data=X_imputed, columns=df.columns, index=df.index)


class ImputerRegressor(_Imputer):
    """
    This class implements a regression imputer in the multivariate case.
    It imputes each column using a single fit-predict for a given estimator, based on the colunms
    which have no missing values.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    estimator : BaseEstimator, optional
        Estimator for imputing a column based on the others
    handler_nan : str
        Can be `fit, `row` or `column`:
        - if `fit`, the estimator is assumed to be robust to missing values
        - if `row` all non complete rows will be removed from the train dataset, and will not be
        used for the inferance,
        - if `column` all non complete columns will be ignored.
        By default, `row`
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations import imputers
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> imputer = imputers.ImputerRegressor(estimator=ExtraTreesRegressor())
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    ...                        [np.nan, np.nan, np.nan, np.nan],
    ...                        [1, 2, 2, 5],
    ...                        [2, 2, 2, 2]],
    ...                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(df)
       var1  var2  var3  var4
    0   1.0   1.0   1.0   1.0
    1   1.0   2.0   2.0   2.0
    2   1.0   2.0   2.0   5.0
    3   2.0   2.0   2.0   2.0
    """

    def __init__(
        self,
        imputer_params: Tuple[str, ...] = ("handler_nan",),
        groups: Tuple[str, ...] = (),
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "row",
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__(
            imputer_params=imputer_params,
            groups=groups,
            random_state=random_state,
        )
        self.estimator = estimator
        self.handler_nan = handler_nan

    def _fit_estimator(self, estimator, X, y) -> Any:
        return estimator.fit(X, y)

    def _predict_estimator(self, estimator, X) -> pd.Series:
        pred = estimator.predict(X)
        return pd.Series(pred, index=X.index)

    def get_Xy_valid(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=col, errors="ignore")
        if self.handler_nan == "none":
            pass
        elif self.handler_nan == "row":
            X = X.loc[~X.isna().any(axis=1)]
        elif self.handler_nan == "column":
            X = X.dropna(how="any", axis=1)
        else:
            raise ValueError(
                f"Value '{self.handler_nan}' is not correct for argument `handler_nan'"
            )
        # X = pd.get_dummies(X, prefix_sep="=")
        y = df.loc[X.index, col]
        return X, y

    def _fit_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> Optional[BaseEstimator]:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return a fitted regressor

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        assert col == "__all__"
        cols_with_nans = df.columns[df.isna().any()]
        dict_estimators: Dict[str, BaseEstimator] = dict()
        for col in cols_with_nans:
            # Selects only the valid values in the Train Set according to the chosen method
            X, y = self.get_Xy_valid(df, col)

            # Selects only non-NaN values for the Test Set
            is_na = y.isna()
            X = X[~is_na]
            y = y[~is_na]

            # Train the model according to an ML or DL method and after predict the imputation
            if not X.empty:
                estimator = copy.deepcopy(self.estimator)
                dict_estimators[col] = self._fit_estimator(estimator, X, y)
            else:
                dict_estimators[col] = None
        return dict_estimators

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        assert col == "__all__"

        df_imputed = df.copy()
        cols_with_nans = df.columns[df.isna().any()]
        for col in cols_with_nans:
            model = self._dict_fitting["__all__"][ngroup][col]
            if model is None:
                continue
            # Define the Train and Test set
            X, y = self.get_Xy_valid(df, col)

            # Selects only non-NaN values for the Test Set
            is_na = y.isna()
            if not np.any(is_na):
                continue
            X = X.loc[is_na]

            y_hat = self._predict_estimator(model, X)
            y_hat.index = X.index
            df_imputed.loc[X.index, col] = y_hat
        return df_imputed


class ImputerRpcaPcp(_Imputer):
    """
    This class implements the Robust Principal Component Analysis imputation with Principal
    Component Pursuit. The imputation minimizes a loss function combining a low-rank criterium on
    the dataframe and a L1 penalization on the residuals.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    columnwise : bool
        For the RPCA method to be applied columnwise (with reshaping of
        each column into an array)
        or to be applied directly on the dataframe. By default, the value is set to False.
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        columnwise: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        period: int = 1,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
        max_iterations: int = int(1e4),
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            imputer_params=(
                "period",
                "mu",
                "lam",
                "max_iterations",
                "tolerance",
            ),
            groups=groups,
            columnwise=columnwise,
            random_state=random_state,
        )

        self.period = period
        self.mu = mu
        self.lam = lam
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

    def get_model(self, **hyperparams) -> rpca_pcp.RpcaPcp:
        """
        Get the underlying model of the imputer based on its attributes.

        Returns
        -------
        rpca.RPCA
            RPCA model to be used in the fit and transform methods.
        """
        hyperparams = {
            key: hyperparams[key]
            for key in [
                "mu",
                "lam",
                "max_iterations",
                "tolerance",
            ]
        }
        model = rpca_pcp.RpcaPcp(random_state=self._rng, verbose=self.verbose, **hyperparams)

        return model

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams()
        model = self.get_model(**hyperparams)

        X = df.astype(float).values

        D = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D)
        # D = utils.linear_interpolation(D)

        means = np.nanmean(D, axis=0)
        stds = np.nanstd(D, axis=0)
        stds = np.where(stds, stds, 1)
        D_scale = (D - means) / stds
        M, A = model.decompose(D_scale, Omega)
        M = M * stds + means
        A = A * stds + means

        M_final = utils.get_shape_original(M, X.shape)
        A_final = utils.get_shape_original(A, X.shape)
        X_imputed = M_final + A_final

        df_imputed = pd.DataFrame(X_imputed, index=df.index, columns=df.columns)
        df_imputed = df.where(~df.isna(), df_imputed)

        return df_imputed


class ImputerRpcaNoisy(_Imputer):
    """
    This class implements the Robust Principal Component Analysis imputation with added noise.
    The imputation minimizes a loss function combining a low-rank criterium on the dataframe and
    a L1 penalization on the residuals.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    columnwise : bool
        For the RPCA method to be applied columnwise (with reshaping of
        each column into an array)
        or to be applied directly on the dataframe. By default, the value is set to False.
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        columnwise: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        period: int = 1,
        mu: Optional[float] = None,
        rank: Optional[int] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        list_periods: Tuple[int, ...] = (),
        list_etas: Tuple[float, ...] = (),
        max_iterations: int = int(1e4),
        tolerance: float = 1e-6,
        norm: Optional[str] = "L2",
        verbose: bool = False,
    ) -> None:
        super().__init__(
            imputer_params=(
                "period",
                "mu",
                "rank",
                "tau",
                "lam",
                "list_periods",
                "list_etas",
                "max_iterations",
                "tolerance",
                "norm",
            ),
            groups=groups,
            columnwise=columnwise,
            random_state=random_state,
        )

        self.period = period
        self.mu = mu
        self.rank = rank
        self.tau = tau
        self.lam = lam
        self.list_periods = list_periods
        self.list_etas = list_etas
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.norm = norm
        self.verbose = verbose

    def get_model(self, **hyperparams) -> rpca_noisy.RpcaNoisy:
        """
        Get the underlying model of the imputer based on its attributes.

        Returns
        -------
        rpca.RPCA
            RPCA model to be used in the fit and transform methods.
        """

        hyperparams = {
            key: hyperparams[key]
            for key in [
                "rank",
                "tau",
                "lam",
                "list_periods",
                "list_etas",
                "max_iterations",
                "tolerance",
                "norm",
            ]
        }
        model = rpca_noisy.RpcaNoisy(random_state=self._rng, verbose=self.verbose, **hyperparams)
        return model

    def _fit_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Fits the imputer on `df`, at the group and/or column level depending on self.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Tuple
            A tuple made of:
            - the reduced decomposition basis
            - the estimated mean of the columns
            - the estimated standard deviation of the columns

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams()
        model = self.get_model(**hyperparams)

        X = df.astype(float).values
        D = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D)
        # D = utils.linear_interpolation(D)

        means = np.nanmean(D, axis=0)
        stds = np.nanstd(D, axis=0)
        stds = np.where(stds, stds, 1)
        D_scale = (D - means) / stds
        _, _, _, Q = model.decompose_with_basis(D_scale, Omega)

        return Q, means, stds

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams()
        model = self.get_model(**hyperparams)

        X = df.astype(float).values

        D = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D)
        # D = utils.linear_interpolation(D)

        Q, means, stds = self._dict_fitting[col][ngroup]

        D_scale = (D - means) / stds
        M, A = model.decompose_on_basis(D_scale, Omega, Q)
        M = M * stds + means
        A = A * stds + means

        M_final = utils.get_shape_original(M, X.shape)

        df_imputed = pd.DataFrame(M_final, index=df.index, columns=df.columns)
        df_imputed = df.where(~df.isna(), df_imputed)

        return df_imputed


class ImputerSoftImpute(_Imputer):
    """
    This class implements the Soft Impute method:

    Hastie, Trevor, et al. Matrix completion and low-rank SVD via fast alternating least squares.
    The Journal of Machine Learning Research 16.1 (2015): 3367-3402.

    This imputation technique is less robust than the RPCA, although it can provide faster.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    columnwise : bool
        For the RPCA method to be applied columnwise (with reshaping of
        each column into an array)
        or to be applied directly on the dataframe. By default, the value is set to False.
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        columnwise: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        period: int = 1,
        rank: Optional[int] = None,
        tolerance: float = 1e-05,
        tau: Optional[float] = None,
        max_iterations: int = 100,
        verbose: bool = False,
    ):
        super().__init__(
            imputer_params=(
                "period",
                "rank",
                "tolerance",
                "tau",
                "max_iterations",
                "verbose",
            ),
            groups=groups,
            columnwise=columnwise,
            random_state=random_state,
        )
        self.period = period
        self.rank = rank
        self.tolerance = tolerance
        self.tau = tau
        self.max_iterations = max_iterations
        self.verbose = verbose

    def get_model(self, **hyperparams) -> softimpute.SoftImpute:
        """
        Get the underlying model of the imputer based on its attributes.

        Returns
        -------
        softimpute.SoftImpute
            Soft Impute model to be used in the transform method.
        """
        hyperparams = {
            key: hyperparams[key]
            for key in [
                "tau",
                "max_iterations",
                "tolerance",
            ]
        }
        model = softimpute.SoftImpute(random_state=self._rng, verbose=self.verbose, **hyperparams)

        return model

    # def _fit_element(
    #     self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    # ) -> softimpute.SoftImpute:
    #     """
    #     Fits the imputer on `df`, at the group and/or column level depending on
    #     self.groups and self.columnwise.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Dataframe on which the imputer is fitted
    #     col : str, optional
    #         Column on which the imputer is fitted, by default "__all__"
    #     ngroup : int, optional
    #         Id of the group on which the method is applied

    #     Returns
    #     -------
    #     Any
    #         Return fitted SoftImpute model

    #     Raises
    #     ------
    #     NotDataFrame
    #         Input has to be a pandas.DataFrame.
    #     """
    #     self._check_dataframe(df)
    #     assert col == "__all__"
    #     hyperparams = self.get_hyperparams()
    #     model = softimpute.SoftImpute(random_state=self._rng, **hyperparams)
    #     model = model.fit(df.values)
    #     return model

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams()
        model = self.get_model(**hyperparams)

        X = df.astype(float).values

        D = utils.prepare_data(X, self.period)
        Omega = ~np.isnan(D)

        M, A = model.decompose(D, Omega)

        M_final = utils.get_shape_original(M, X.shape)
        A_final = utils.get_shape_original(A, X.shape)
        X_imputed = M_final + A_final

        df_imputed = pd.DataFrame(X_imputed, index=df.index, columns=df.columns)
        df_imputed = df.where(~df.isna(), df_imputed)

        return df_imputed

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_fit2d_1sample": "This test shouldn't be running at all!",
                "check_fit2d_1feature": "This test shouldn't be running at all!",
            },
        }


class ImputerEM(_Imputer):
    """
    This class implements an imputation method based on joint modelling and an inference using a
    Expectation-Minimization algorithm.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    method : {'multinormal', 'VAR'}, default='multinormal'
        Method defining the hypothesis made on the data distribution. Possible values:
        - 'multinormal' : the data points a independent and uniformly distributed following a
        multinormal distribution
        - 'VAR' : the data is a time series modeled by a VAR(p) process
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
        groups: Tuple[str, ...] = (),
        model: Optional[str] = "multinormal",
        columnwise: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        method: Literal["mle", "sample"] = "sample",
        max_iter_em: int = 200,
        n_iter_ou: int = 50,
        ampli: float = 1,
        dt: float = 2e-2,
        tolerance: float = 1e-4,
        stagnation_threshold: float = 5e-3,
        stagnation_loglik: float = 2,
        period: int = 1,
        verbose: bool = False,
        p: Union[None, int] = None,
    ):
        super().__init__(
            imputer_params=(
                "max_iter_em",
                "n_iter_ou",
                "ampli",
                "dt",
                "tolerance",
                "stagnation_threshold",
                "stagnation_loglik",
                "period",
                "p",
            ),
            groups=groups,
            columnwise=columnwise,
            random_state=random_state,
        )
        self.model = model
        self.method = method
        self.max_iter_em = max_iter_em
        self.n_iter_ou = n_iter_ou
        self.ampli = ampli
        self.dt = dt
        self.tolerance = tolerance
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_loglik = stagnation_loglik
        self.period = period
        self.verbose = verbose
        self.p = p

    def get_model(self, **hyperparams) -> em_sampler.EM:
        """Get the underlying model of the imputer based on its attributes.

        Returns
        -------
        em_sampler.EM
            EM model to be used in the fit and transform methods.
        """
        if self.model == "multinormal":
            hyperparams.pop("p")
            return em_sampler.MultiNormalEM(
                random_state=self.random_state,
                method=self.method,
                verbose=self.verbose,
                **hyperparams,
            )
        elif self.model == "VAR":
            hyperparams["p"] = self.p
            return em_sampler.VARpEM(
                random_state=self.random_state,
                method=self.method,
                verbose=self.verbose,
                **(hyperparams),  # type: ignore #noqa
            )
        else:
            raise ValueError(
                f"Model argument `{self.model}` is invalid!"
                " Valid values are `multinormal`and `VAR`."
            )

    def _fit_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> em_sampler.EM:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted EM model

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hyperparams = self.get_hyperparams()
        model = self.get_model(**hyperparams)
        model = model.fit(df.values)
        return model

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe.

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)

        if df.notna().all().all():
            return df
        model = self._dict_fitting[col][ngroup]

        X = df.values.astype(float)
        X_imputed = model.transform(X)

        df_transformed = pd.DataFrame(X_imputed, columns=df.columns, index=df.index)

        return df_transformed
