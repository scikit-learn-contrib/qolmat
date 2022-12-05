from __future__ import annotations

from functools import reduce
import logging
from typing import Optional
from warnings import WarningMessage

import numpy as np
import pandas as pd
import scipy
from numpy import linalg as nl
from numpy.typing import ArrayLike
from sklearn.impute._base import _BaseImputer
import sys

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
        strategy: Optional[str] = "argmax",
        n_iter_em: Optional[int] = 7,
        n_iter_ou: Optional[int] = 50,
        ampli: Optional[int] = 0.5,
        random_state: Optional[int] = 123,
        verbose: Optional[int] = 0,
        temporal: Optional[bool] = True,
        dt: Optional[float] = 2e-2,
        convergence_threshold: Optional[float] = 1e-6,
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
        self.dt = dt
        self.convergence_threshold = convergence_threshold

    def _linear_interpolation(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing data with a linear interpolation, column-wise

        Parameters
        ----------
        X : np.ndarray
            array with missing values

        Returns
        -------
        X_interpolated : np.ndarray
            imputed array, by linear interpolation
        """
        X_interpolated = X.copy()
        nans, x = np.isnan(X_interpolated), lambda z: z.nonzero()[0]
        X_interpolated[nans] = np.interp(x(nans), x(~nans), X_interpolated[~nans])
        return X_interpolated

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

    def _check_convergence(
        self,
        mu: np.ndarray,
        mu_prec: np.ndarray,
        cov: np.ndarray,
        cov_prec: np.ndarray,
    ) -> bool:
        """Check if the EM algorithm has converged

        Parameters
        ----------
        mu : np.ndarray
            the actual value of the mean
        mu_prec : np.ndarray
            the previous value of the mean
        cov : np.ndarray
            the actual value of the covariance
        cov_prec : np.ndarray
            the previous value of the covariance

        Returns
        -------
        bool
            True/False if the algorithm has converged
        """
        return (
            np.linalg.norm(mu - mu_prec) < self.convergence_threshold
            and np.linalg.norm(cov - cov_prec, ord=2) < self.convergence_threshold
        )

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
            X_ystd = np.roll(X, 1, axis=0).astype("float")
            X_ystd[:1, :] = np.nan
            X_shifted = np.hstack([X_shifted, X_ystd])
        if tmrw:
            X_tmrw = np.roll(X, -1, axis=0).astype("float")
            X_tmrw[-1:, :] = np.nan
            X_shifted = np.hstack([X_shifted, X_tmrw])
        return X_shifted

    def _em(
        self, X: np.ndarray, observed: np.ndarray, missing: np.ndarray
    ) -> np.ndarray:
        """EM step

        Parameters
        ----------
        X : np.ndarray
            array with the current imputations
        observed : np.ndarray
            mask of the observed values
        missing : np.ndarray
            mask of the missing values

        Returns
        -------
        np.ndarray
            new imputed dataset
        """

        nr, nc = X.shape

        one_to_nc = np.arange(1, nc + 1, step=1)
        mu_tilde, sigma_tilde = {}, {}

        for i in range(nr):
            sigma_tilde[i] = np.zeros(nc**2).reshape(nc, nc)
            if set(observed[i, :]) == set(one_to_nc - 1):
                continue

            missing_i = missing[i, :][missing[i, :] != -1]
            observed_i = observed[i, :][observed[i, :] != -1]

            sigma_MM = self.cov[np.ix_(missing_i, missing_i)]
            sigma_MO = self.cov[np.ix_(missing_i, observed_i)]
            sigma_OM = sigma_MO.T
            sigma_OO = self.cov[np.ix_(observed_i, observed_i)]

            mu_tilde[i] = self.means[np.ix_(missing_i)] + sigma_MO @ np.linalg.inv(
                sigma_OO
            ) @ (X[i, observed_i] - self.means[np.ix_(observed_i)])
            sigma_MM_O = sigma_MM - sigma_MO @ np.linalg.inv(sigma_OO) @ sigma_OM
            sigma_tilde[i][np.ix_(missing_i, missing_i)] = sigma_MM_O

            if self.strategy == "argmax":
                X[i, missing_i] = mu_tilde[i]
            if self.strategy == "sample":
                X[i, missing_i] = np.random.multivariate_normal(mu_tilde[i], sigma_MM_O)

        self.means = np.mean(X, axis=0)
        self.cov = np.cov(X.T, bias=1) + reduce(np.add, sigma_tilde.values()) / nr

        return X

    def impute_em(self, X: ArrayLike) -> ArrayLike:
        """Imputation via EM algorithm

        Parameters
        ----------
        X : ArrayLike
            array with missing values

        Returns
        -------
        X_transformed
            imputed array
        """

        X_ = self._convert_numpy(X)
        if np.nansum(X_) == 0:
            return X_

        _, nc_init = X_.shape

        if self.temporal:
            X_ = self._add_shift(X_, ystd=True, tmrw=True)
            _, nc = X_.shape

        mask = ~np.isnan(X_)
        one_to_nc = np.arange(1, nc + 1, step=1)
        observed = one_to_nc * (~mask) - 1
        missing = one_to_nc * mask - 1

        # first imputation
        X_transformed = np.apply_along_axis(self._linear_interpolation, 0, X_)

        # first estimation of params
        self.means = np.mean(X_transformed, axis=0)
        self.cov = np.cov(X_transformed.T)
        # In case of inversibility problem, one can add a penalty term
        eps = 1e-2
        self.cov -= eps * (self.cov - np.diag(self.cov.diagonal()))
        if scipy.linalg.eigh(self.cov)[0].min() < 0:
            raise WarningMessage(
                f"Negative eigenvalue, some variables may be constant or colinear, "
                f"min value of {scipy.linalg.eigh(self.cov)[0].min():.3g} found."
            )
        if np.abs(np.linalg.eigh(self.cov)[0].min()) > 1e20:
            raise WarningMessage("Large eigenvalues, imputation may be inflated.")
        self.inv_cov = nl.pinv(self.cov)

        for _ in range(self.n_iter_em):
            try:
                X_transformed = self._em(X_transformed, observed, missing)

            except BaseException:
                raise WarningMessage("EM step failed.")

        if self.temporal:
            X_transformed = X_transformed[:, :nc_init]

        if np.all(np.isnan(X_transformed)):
            raise WarningMessage("Result contains NaN. This is a bug.")

        return X_transformed

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

        X_imputed = self.impute_em(X)

        if isinstance(X, np.ndarray):
            return X_imputed
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
        else:
            raise AssertionError(
                "Invalid type. X must be either pd.DataFrame or np.ndarray."
            )
