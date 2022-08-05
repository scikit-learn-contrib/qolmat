from __future__ import annotations

import logging
from typing import Optional
from warnings import WarningMessage

import bottleneck
import numpy as np
import pandas as pd
import scipy
from numpy import linalg as nl
from numpy.typing import ArrayLike
from scipy import linalg as sl
from scipy.ndimage import shift
from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ImputeEM(_BaseImputer):  # type: ignore
    """
    Imputation of missing values using a multivariate Gaussian model through EM optimization and
    using a projected Ornstein-Uhlenbeck process.

    Parameters
    ----------
    method : str
        Method for imputation, choose among "sample" or "argmax".
    n_iter_em : int, optional
        Number of shifts added for temporal memory,
        is equivalent to n_iter (index) of memory padding, by default 14.
    n_iter_ou : int, optional
        Number of iterations for the Gibbs sampling method (+ noise addition),
        necessary for convergence, by default 50.
    ampli : float, optional
        Whether to sample the posterior (1)
        or to maximise likelihood (0), by default 1.
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility.
    verbose : int, optional
        Verbosity flag, controls the debug messages that are issued as functions are evaluated.
        The higher, the more verbose. Can be 0, 1, or 2. By default 0.
    temporal: bool, optional
        if temporal data, extend the matrix to have -1 and +1 shift

    Attributes
    ----------
    X_intermediate : list
        List of pd.DataFrame giving the results of the EM process as function of the
        iteration number.

    """

    def __init__(
        self,
        strategy: Optional[str] = "sample",
        n_iter_em: Optional[int] = 7,
        n_iter_ou: Optional[int] = 50,
        ampli: Optional[int] = 0.5,
        random_state: Optional[int] = 123,
        verbose: Optional[int] = 0,
        temporal: Optional[bool] = True,
    ) -> None:

        self.strategy = strategy
        self.n_iter_em = n_iter_em
        self.n_iter_ou = n_iter_ou
        self.ampli = ampli
        self.random_state = random_state
        self.verbose = verbose
        self.mask_outliers = None
        self.cov = None
        self.temporal = temporal

    def linear_interpolation(self, data):
        """
        Impute missing data with a linear interpolation, column-wise

        Parameters
        ----------
        data : np.ndarray
            array with missing values

        Returns
        -------
        np.ndarray
            imputed array
        """
        interpolated_data = data.copy()
        bad_indexes = np.isnan(interpolated_data)
        good_indexes = np.logical_not(bad_indexes)
        good_data = data[good_indexes]
        interpolated = np.interp(
            bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data
        )
        interpolated_data[bad_indexes] = interpolated
        return interpolated_data

    def _gradient_conjugue(
        self, A: ArrayLike, X: ArrayLike, tol: float = 1e-6
    ) -> ArrayLike:
        """
        Minimize np.sum(X * AX) by imputing missing values.
        To this aim, we compute in parallel a gradient algorithm for each data.

        Parameters
        ----------
        A : ArrayLike
            A array
        X : ArrayLike
            X array
        tol : float, optional
            Tolerance, by default 1e-6

        Returns
        -------
        ArrayLike
            Minimized array.
        """

        index_imputed = np.any(np.isnan(X), axis=1)
        X_temp = X[index_imputed].transpose().copy()
        n_iter = (
            np.max(np.sum(np.isnan(X_temp), axis=0))
            if len(np.sum(np.isnan(X_temp), axis=0)) > 0
            else 10
        )
        n_var, n_ind = X_temp.shape
        mask = np.isnan(X_temp)
        X_temp[mask] = 0
        b = -A @ X_temp
        b[~mask] = 0
        xn, pn, rn = np.zeros(X_temp.shape), b, b  # Initialisation

        for n in range(n_iter + 2):
            Apn = A @ pn
            Apn[~mask] = 0
            alphan = np.sum(rn ** 2, axis=0) / np.sum(pn * Apn, axis=0)
            alphan[
                np.isnan(alphan)
            ] = 0  # we stop updating if convergence is reached for this date
            xn, rnp1 = xn + alphan * pn, rn - alphan * Apn
            betan = np.sum(rnp1 ** 2, axis=0) / np.sum(rn ** 2, axis=0)
            betan[
                np.isnan(betan)
            ] = 0  # we stop updating if convergence is reached for this date
            pn, rn = rnp1 + betan * pn, rnp1

        X_temp[mask] = xn[mask]
        X_final = X.copy()
        X_final[index_imputed] = X_temp.transpose()

        return X_final

    def _add_shift(
        self, X: ArrayLike, ystd: bool = True, tmrw: bool = False
    ) -> ArrayLike:
        """
        For each variable adds one columns corresponding to day before and/or after
        each date (date must be by row index).

        Parameters
        ----------
        X : ArrayLike
            Input array.
        ystd : bool, optional
            Include values from the past, by default True.
        tmrw : bool, optional
            Include values from the future, by default False.

        Returns
        -------
        ArrayLike
            Extended array with shifted variables.
        """

        X_shifted = X.copy()
        if ystd:
            X_ystd = np.c_[X, X[:, -1]]
            X_ystd = shift(X_ystd, 1, cval=np.NaN)[:, 1:]
            X_shifted = np.hstack([X_shifted, X_ystd])
        if tmrw:
            X_tmrw = np.c_[X[:, 0], X]
            X_tmrw = shift(X_tmrw, -1, cval=np.NaN)[:, :-1]
            X_shifted = np.hstack([X_shifted, X_tmrw])
        return X_shifted

    def _argmax_posterior(self, df: ArrayLike) -> ArrayLike:
        """
        Get the argmax of a posterior distribution. Called by `impute_mapem_ts`.

        Parameters
        ----------
        df : ArrayLike
            Input DataFrame.

        Returns
        -------
        ArrayLike
            DataFrame with imputed values.
        """
        df_imputed = self.means + self._gradient_conjugue(self.inv_cov, df - self.means)
        return df_imputed

    def _sample_ou(
        self,
        X: ArrayLike,
        mask_na: ArrayLike,
        rng: int,
        dt: float = 2e-2,
    ) -> ArrayLike:
        """
        Samples the Gaussian distribution under the constraint that not na values must remain
        unchanged, using a projected Ornstein-Uhlenbeck process.
        The sampled distribution tends to the target one in the limit dt -> 0 and
        n_iter_ou x dt -> infty.
        Called by `impute_sample_ts`.

        Parameters
        ----------
        df : ArrayLike
            Inital dataframe to be imputed, which should have been already imputed using a simple
            method. This first imputation will be used as an initial guess.
        mask_na : ArrayLike
            Boolean dataframe indicating which coefficients should be resampled.
        rng : int
            Random number generator to be used (for reproducibility).
        n_iter_ou : int
            Number of iterations for the OU process, a large value decreases the sample bias
            but increases the computation time.
        dt : float
            Process integration time step, a large value increases the sample bias and can make
            the algorithm unstable, but compensates for a smaller n_iter_ou. By default, 2e-2.
        ampli : float
            Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

        Returns
        -------
        ArrayLike
            DataFrame after Ornstein-Uhlenbeck process.
        """
        n_samples, n_variables = X.shape

        X = (X - self.means).copy()
        X_init = X.copy()
        beta = self.cov.copy()
        beta[:] = sl.inv(beta)
        gamma = np.diagonal(self.cov)
        for i in range(self.n_iter_ou):
            noise = self.ampli * rng.normal(0, 1, size=(n_samples, n_variables))
            X += -dt * gamma * (X @ beta) + noise * np.sqrt(2 * gamma * dt)
            X[~mask_na] = X_init[~mask_na]
        return X + self.means

    def _check_X(self, X: ArrayLike) -> pd.DataFrame:
        """
        Convert X array to a pd.DataFrame for internal calculations.

        Parameters
        ----------
        X : ArrayLike
            Input Array.

        Returns
        -------
        pd.DataFrame
            Return DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            self.type_X = type(X)
            if (not isinstance(X, np.ndarray)) & (not isinstance(X, list)):
                raise ValueError(
                    "Input array is not a list, np.array, nor pd.DataFrame."
                )
            X = pd.DataFrame(X)
            X.columns = X.columns.astype(str)
        return X

    def _convert_numpy(self, X: ArrayLike) -> np.ndarray:
        """
        Convert X pd.DataFrame to an array for internal calculations.

        Parameters
        ----------
        X : ArrayLike
            Input Array.

        Returns
        -------
        np.ndarray
            Return Array.
        """
        if not isinstance(X, np.ndarray):
            if (not isinstance(X, pd.DataFrame)) & (not isinstance(X, list)):
                raise ValueError(
                    "Input array is not a list, np.array, nor pd.DataFrame."
                )
            X = X.to_numpy()
        return X

    def fit(self, X: ArrayLike) -> ImputeEM:
        """
        Fits covariance and compute inverted matrix.
        A penalization is applied to avoid singularity.
        Starts off with an interpolation as initialization to be able to compute covariance.

        Parameters
        ----------
        df : ArrayLike
            Sparse array to be imputed.

        Returns
        -------
        ImputeEMNumpy
            The class itself.
        """
        X = self._convert_numpy(X)
        X_nonan = np.apply_along_axis(self.linear_interpolation, 0, X)
        self.means = np.mean(X_nonan, axis=0)
        self.cov = np.cov(X_nonan.T)

        # In case of inversibility problem, one can add a penalty term
        eps = 1e-2
        self.cov -= eps * (self.cov - np.diag(self.cov.diagonal()))

        if scipy.linalg.eigh(self.cov)[0].min() < 0:
            raise WarningMessage(
                f"Negative eigenvalue, some variables may be constant or colinear, "
                f"min value of {scipy.linalg.eigh(self.cov)[0].min():.3g} found."
            )
        if np.abs(np.linalg.eigh(self.cov)[0].min()) > 1e20:
            raise WarningMessage("Large eigenvalues, imputation may be inflated")

        self.inv_cov = nl.pinv(self.cov)
        # self.inv_cov_nodiag = self.inv_cov.copy()
        # self.inv_cov_nodiag = np.fill_diagonal(self.inv_cov_nodiag, 0)
        # self.inv_cov_diag = np.diagonal(self.inv_cov)

        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Imputes every missing value in df via EM (expectation-maximisation) sampling or
        maximum likelihood estimation.

        Parameters
        ----------
        X : ArrayLike
            Sparse array to be imputed.

        Returns
        -------
        ArrayLike
            Final array after EM sampling.
        """

        X_ = self._convert_numpy(X)
        # X_ = X.copy()
        n, m = X.shape

        if np.nansum(X_) == 0:
            return X_
        rng = np.random.default_rng(self.random_state)
        X_ = np.apply_along_axis(self.linear_interpolation, 0, X_)
        if self.temporal:
            mask_na = np.zeros(
                self._add_shift(X_, ystd=True, tmrw=True).shape, dtype=bool
            )
        else:
            mask_na = np.zeros(X_.shape, dtype=bool)
        mask_na[:, :m] = np.isnan(X)
        scaler = StandardScaler()
        X_ = scaler.fit_transform(X_)
        X_intermediate_ = []
        for i in range(self.n_iter_em):
            if self.temporal:
                X_ = self._add_shift(X_, ystd=True, tmrw=True)
                X_ = bottleneck.push(X_, axis=0)
                X_ = bottleneck.push(X_[::-1], axis=0)[::-1]
            self.fit(X_)
            if self.strategy == "sample":
                X_ = self._sample_ou(X_, mask_na, rng, dt=2e-2)
            elif self.strategy == "argmax":
                X_ = self._argmax_posterior(X_)
            else:
                raise AssertionError(
                    "Invalid 'method' argument. Choose among 'argmax' or 'sample'."
                )
            if self.temporal:
                X_ = X_[:, :m]
            X_intermediate_.append(X_.copy())
        X_ = scaler.inverse_transform(X_)
        self.X_intermediate = [
            scaler.inverse_transform(X_inter_) for X_inter_ in X_intermediate_
        ]
        self.fit(X_)

        if isinstance(X, np.ndarray):
            return X_
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_, index=X.index, columns=X.columns)

        return X_

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit and impute input X array.

        Parameters
        ----------
        X : ArrayLike
            Sparse array to be imputed.

        Returns
        -------
        ArrayLike
            Final array after EM sampling.
        """
        if not ((isinstance(X, np.ndarray)) or (isinstance(X, pd.DataFrame))):
            raise AssertionError(
                "Invalid type. X must be either pd.DataFrame or np.ndarray."
            )
        self.fit(X)
        return self.transform(X)
