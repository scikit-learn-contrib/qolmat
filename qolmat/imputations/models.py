from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from qolmat.imputations.rpca.pcp_rpca import RPCA
from qolmat.imputations.rpca.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA
from qolmat.benchmark import utils
import os
import sys


class ImputeColumnWise:
    """
    This class implements the imputation column wise.

    Parameters
    ----------
    groups : Optional[List]
        names for the custom groupby
    """

    def __init__(
        self,
        groups=[],
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
        col_to_impute = df.columns
        imputed = df.copy()
        for col in col_to_impute:
            imputed[col] = self.fit_transform_col(df[col]).values
        imputed.fillna(0, inplace=True)
        return imputed

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
    """

    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform the Imputer to the dataset by fitting with the mean of each column

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
            utils.custom_groupby(signal, self.groups)[col].transform("mean")
        )
        return imputed


class ImputeByMedian(ImputeColumnWise):
    """
    This class implements the implementation by the median of each column
    """

    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

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
            utils.custom_groupby(signal, self.groups)[col].transform("median")
        )
        return imputed


class ImputeByMode(ImputeColumnWise):
    """
    This class implements the implementation by the mode of each column
    """

    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform the Imputer to the dataset by fitting with the mode of each column

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
            utils.custom_groupby(signal, self.groups)[col].mode().iloc[0]
        )
        return imputed


class RandomImpute(ImputeColumnWise):
    """
    This class implements the imputation by a random value (from th eobserved ones) of each column
    """

    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform the Imputer to the dataset by fitting with a random value

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
        imputed = signal.reset_index()
        number_missing = imputed[col].isnull().sum()
        obs = imputed.loc[imputed[col].notnull(), col].values
        imputed.loc[imputed[col].isnull(), col] = np.random.choice(
            obs, number_missing, replace=True
        )
        return imputed[col]


class ImputeLOCF(ImputeColumnWise):
    """
    This class implements a forward imputation, column wise
    """

    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform by imputing missing values by carrying the last observation forward.
        If the first observation is missing, it is imputed by the median of the series

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
        imputed = signal.reset_index()
        imputed = utils.custom_groupby(imputed, self.groups)[col].transform(
            lambda x: x.ffill()
        )
        return imputed.fillna(np.nanmedian(imputed))


class ImputeNOCB(ImputeColumnWise):
    """
    This class implements a backawrd imputation, column wise
    """

    def __init__(
        self,
        groups=[],
    ) -> None:
        super().__init__(groups=groups)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform by imputing missing values by carrying carrying the next observation backward.
        If the last observation is missing, it is imputed by the median of the series

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
        imputed = signal.reset_index()
        imputed = utils.custom_groupby(imputed, self.groups)[col].transform(
            lambda x: x.bfill()
        )
        return imputed.fillna(np.nanmedian(imputed))


class ImputeKNN(ImputeColumnWise):
    """
    This class implements an imputation by the k-nearest neighbors, column wise

    Parameters
    ----------
    k : int
        number of nearest neighbors
    """

    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform by imputing missing values with the KNN method.

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
        imputed = np.asarray(signal[col]).reshape(-1, 1)
        imputer = KNNImputer(n_neighbors=self.k)
        imputed = imputer.fit_transform(imputed)
        imputed = pd.Series([a[0] for a in imputed], index=signal.index)
        return imputed.fillna(np.nanmedian(imputed))

    def get_hyperparams(self) -> Dict[str, int]:
        """
        Get the value of the hyperparameter of the method, i.e. number of nearest neighbors

        Returns
        -------
        Dict[str, int]
            number of nearest neighbors
        """
        return {"k": self.k}


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
        By default, the value is set to "linear"
    order : Optional[int]
        order for the spline interpolation
    """

    def __init__(self, **kwargs) -> None:
        self.method = "linear"
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform missing values using interpolation techniques from  pd.Series.interpolat

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
        signal = signal.set_index("datetime")
        if self.method in ["spline", "polynomial"]:
            return signal[col].interpolate(method=self.method, order=self.order)
        return signal[col].interpolate(method=self.method)


class ImputeBySpline(ImputeColumnWise):
    """
    This class implements a way to impute using splines.
    Note : overlapping with ImputeByInterpolation since same function but with other parameters ..?
    """

    def __init__(
        self,
    ) -> None:
        pass

    def fit_transform_col(self, signal: pd.Series) -> pd.Series:
        """
        Fit/transform missing values using splines.

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
        signal = signal.set_index("datetime")
        return signal[col].interpolate(option="spline")


class ImputeRPCA:
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

    def __init__(self, method, multivariate=False, **kwargs) -> None:
        self.multivariate = multivariate
        self.method = method

        if method == "PCP":
            self.rpca = RPCA()
        elif method == "temporal":
            self.rpca = TemporalRPCA()
        elif method == "online":
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
            for col in df.columns:
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


class ImputeIterative:
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
        iterative_imputer = IterativeImputer(**(self.kwargs))
        res = iterative_imputer.fit_transform(df.values)
        imputed = pd.DataFrame(columns=df.columns)
        for ind, col in enumerate(imputed.columns):
            imputed[col] = res[:, ind]
        imputed.index = df.index
        return imputed

    def get_hyperparams(self):
        return self.__dict__["kwargs"]


class ImputeRegressor:
    """
    This class implements a regression imputer in the multivariate case.
    It imputes each Series with missign value within a DataFrame using the complete ones.

    Parameters
    ----------
    model :
        regression model
    """

    def __init__(self, model, cols_to_impute=[], **kwargs) -> None:
        self.model = model
        self.cols_to_impute = cols_to_impute
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
        df_imp = df.copy()
        if self.cols_to_impute == []:
            self.cols_to_impute = df.columns.tolist()
        X = df[[col for col in df.columns if col not in self.cols_to_impute]]
        for col in self.cols_to_impute:
            y = df[col]
            is_na = y.isna()
            self.model.fit(X[~is_na], y[~is_na])
            df_imp.loc[is_na, col] = self.model.predict(X[is_na])

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


class ImputeStochasticRegressor:
    """
    This class implements a stochastic regression imputer in the multivariate case.
    It imputes each Series with missign value within a DataFrame using the complete ones.

    Parameters
    ----------
    model :
        regression model
    """

    def __init__(self, model, cols_to_impute=[], **kwargs) -> None:
        self.model = model
        self.cols_to_impute = cols_to_impute
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
        if self.cols_to_impute == []:
            self.cols_to_impute = df.columns.tolist()
        X = df[[col for col in df.columns if col not in self.cols_to_impute]]
        for col in self.cols_to_impute:
            y = df[col]
            is_na = y.isna()
            self.model.fit(X[~is_na], y[~is_na])
            y_pred = self.model.predict(X)
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
