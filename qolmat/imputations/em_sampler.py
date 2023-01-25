from __future__ import annotations

import logging
from functools import reduce
from typing import List, Optional
from warnings import WarningMessage

import numpy as np
import pandas as pd
import scipy
from numpy.typing import ArrayLike
from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _gradient_conjugue(
        A: ArrayLike, X: ArrayLike, tol: float = 1e-6
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
    index_imputed = np.isnan(X).any(axis=1)
    X_temp = X[index_imputed, :].transpose().copy()
    n_iter = np.isnan(X_temp).sum(axis=0).max()
    mask = np.isnan(X_temp)
    X_temp[mask] = 0
    b = -A @ X_temp
    b[~mask] = 0
    xn, pn, rn = np.zeros(X_temp.shape), b, b  # Initialisation
    for n in range(n_iter + 2):
        # if np.max(np.sum(rn**2)) < tol : # Condition de sortie " usuelle "
        #     X_temp[mask] = xn[mask]
        #     return X_temp.transpose()
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


class ImputeEM(_BaseImputer):  # type: ignore
    """
    Imputation of missing values using a multivariate Gaussian model through EM optimization and
    using a projected Ornstein-Uhlenbeck process.

    Parameters
    ----------
    method : str
        Method for imputation, choose among "sample" or "mle".
    max_iter_em : int, optional
        Maximum number of steps in the EM algorithm
    n_iter_ou : int, optional
        Number of iterations for the Gibbs sampling method (+ noise addition),
        necessary for convergence, by default 50.
    ampli : float, optional
        Whether to sample the posterior (1)
        or to maximise likelihood (0), by default 1.
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility.
    verbose : bool, optional
        Verbosity flag, controls the debug messages that are issued as functions are evaluated.
    temporal: bool, optional
        if temporal data, extend the matrix to have -1 and +1 shift

    Attributes
    ----------
    X_intermediate : list
        List of pd.DataFrame giving the results of the EM process as function of the
        iteration number.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.em_sampler import ImputeEM
    >>> imputor = ImputeEM(strategy="sample")
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 3, 2],
    >>>                        [1, 2, 2, 1], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        strategy: Optional[str] = "mle",
        max_iter_em: Optional[int] = 200,
        n_iter_ou: Optional[int] = 50,
        ampli: Optional[int] = 1,
        random_state: Optional[int] = 123,
        verbose: Optional[bool] = True,
        temporal: Optional[bool] = False,
        dt: Optional[float] = 2e-2,
        tolerance: Optional[float] = 1e-4,
        stagnation_threshold: Optional[float] = 5e-3,
        stagnation_loglik: Optional[float] = 1e1,
    ) -> None:

        if strategy not in ["mle", "ou"]:
            raise Exception("strategy has to be 'mle' or 'ou'")

        self.strategy = strategy
        self.max_iter_em = max_iter_em
        self.n_iter_ou = n_iter_ou
        self.ampli = ampli
        self.random_state = random_state
        self.verbose = verbose
        self.mask_outliers = None
        self.cov = None
        self.temporal = temporal
        self.dt = dt
        self.convergence_threshold = tolerance
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_loglik = stagnation_loglik

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
                raise ValueError("Input array is not a list, np.array, nor pd.DataFrame.")
            X = X.to_numpy()
        return X

    def _check_convergence(
        self,
        imputations: List[np.ndarray],
        mu: List[np.ndarray],
        cov: List[np.ndarray],
        n_iter: int,
    ) -> bool:
        """Check if the EM algorithm has converged. Three criteria:
        1) if the differences between the estimates of the parameters (mean and covariance) is
        less than a threshold (min_diff_reached - convergence_threshold).
        2) if the difference of the consecutive differences of the estimates is less than a
        threshold, i.e. stagnates over the last 5 interactions (min_diff_stable -
        stagnation_threshold).
        3) if the likelihood of the data no longer increases,
        i.e. stagnates over the last 5 iterations (max_loglik - stagnation_loglik).

        Parameters
        ----------
        imputations : List[np.ndarray]
            list of imputations estimates
        mu : List[np.ndarray]
            list of mean estimates
        cov : List[np.ndarray]
            list of covariance estimates
        n_iter: int,
            current iteration

        Returns
        -------
        bool
            True/False if the algorithm has converged
        """

        min_diff_reached = (
            n_iter > 5
            and scipy.linalg.norm(mu[-1] - mu[-2], np.inf) < self.convergence_threshold
            and scipy.linalg.norm(cov[-1] - cov[-2], np.inf) < self.convergence_threshold
        )

        min_diff_stable = (
            n_iter > 10
            and min([scipy.linalg.norm(t - s, np.inf) for s, t in zip(mu[-6:], mu[-5:])])
            < self.stagnation_threshold
            and min([scipy.linalg.norm(t - s, np.inf) for s, t in zip(cov[-6:], cov[-5:])])
            < self.stagnation_threshold
        )

        if n_iter > 10:
            logliks = [
                scipy.stats.multivariate_normal.logpdf(data, m, c).sum()
                for data, m, c in zip(imputations[-6:], mu[-6:], cov[-6:])
            ]
        max_loglik = n_iter > 10 and min(abs(pd.Series(logliks).diff())) < self.stagnation_loglik

        return min_diff_reached or min_diff_stable or max_loglik
    
    def _invert_covariance(self, epsilon=1e-2):
        # In case of inversibility problem, one can add a penalty term
        cov = self.cov - epsilon * (self.cov - np.diag(self.cov.diagonal()))
        if scipy.linalg.eigh(self.cov)[0].min() < 0:
            raise WarningMessage(
                f"Negative eigenvalue, some variables may be constant or colinear, "
                f"min value of {scipy.linalg.eigh(self.cov)[0].min():.3g} found."
            )
        if np.abs(scipy.linalg.eigh(self.cov)[0].min()) > 1e20:
            raise WarningMessage("Large eigenvalues, imputation may be inflated.")
        
        return scipy.linalg.inv(cov)

    def _add_shift(self, X: ArrayLike, ystd: bool = True, tmrw: bool = False) -> ArrayLike:
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

        X_init = X.copy()
        beta = self.cov_inv
        gamma = np.diagonal(self.cov)
        # list_X = []
        list_mu = []
        list_cov = []
        for iter_ou in range(self.n_iter_ou):
            noise = self.ampli * rng.normal(0, 1, size=(n_samples, n_variables))
            X += gamma * ((self.means - X) @ beta) * dt + noise * np.sqrt(2 * gamma * dt)
            X[~mask_na] = X_init[~mask_na]
            if iter_ou > self.n_iter_ou - 50:
                # list_X.append(X)
                list_mu.append(np.mean(X, axis=0))
                list_cov.append(np.cov(X.T, bias=True))

        # MANOVA formula
        mu_stack = np.stack(list_mu, axis=1)
        self.mu = np.mean(mu_stack, axis=1)
        cov_stack = np.stack(list_cov, axis=2)
        self.cov = np.mean(cov_stack, axis=2) + np.cov(mu_stack, bias=True)
        self.cov_inv = self._invert_covariance(epsilon=1e-2)

        return X
    
    def _maximize_likelihood(self, X: ArrayLike) -> ArrayLike:
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
        X_imputed = self.means + _gradient_conjugue(self.cov_inv, X - self.means)
        return X_imputed

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

        rng = np.random.default_rng(self.random_state)
        _, nc = X_.shape

        mask_na = np.isnan(X)
            

        # first imputation
        X_transformed = np.apply_along_axis(self._linear_interpolation, 0, X_)

        # first estimation of params
        self.means = np.mean(X_transformed, axis=0)
        self.cov = np.cov(X_transformed.T)
        
        self.cov_inv = self._invert_covariance(epsilon=1e-2)

        if self.temporal:
            shift = 1
            X_past = X_transformed.copy()
            X_past = np.roll(X_past, shift, axis=0).astype(float)
            X_past[:shift, :] = np.nan

        list_X_transformed, list_means, list_covs = [], [], []
        for iter_em in range(self.max_iter_em):

            if self.temporal:
                shift = 1
                X_past = X_transformed.copy()
                X_past = np.roll(X_past, shift, axis=0).astype(float)
                X_past[:shift, :] = np.nan

            X_transformed = self._sample_ou(X_transformed, mask_na, rng, dt=2e-2)

            list_means.append(self.means)
            list_covs.append(self.cov)
            list_X_transformed.append(X_transformed)
            if (
                self._check_convergence(list_X_transformed, list_means, list_covs, iter_em)
                and self.verbose
            ):
                print(f"EM converged after {iter_em} iterations.")
                break
            
        if self.strategy == "mle":
            X_transformed = self._maximize_likelihood(X_)
        elif self.strategy == "ou":
            X_transformed = self._sample_ou(X_transformed, mask_na, rng, dt=2e-2)

        if np.all(np.isnan(X_transformed)):
            raise WarningMessage("Result contains NaN. This is a bug.")

        self.delta_means = [
            scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_means, list_means[1:])
        ]
        self.delta_covs = [
            scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_covs, list_covs[1:])
        ]
        self.logliks = [
            scipy.stats.multivariate_normal.logpdf(data, m, c).sum()
            for data, m, c in zip(list_X_transformed, list_means, list_covs)
        ]

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
            raise AssertionError("Invalid type. X must be either pd.DataFrame or np.ndarray.")

        if X.shape[1] < 2:
            raise AssertionError("Invalid dimensions: X must be of dimension (n,m) with m>1.")

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        X_imputed = self.impute_em(X_sc)
        X_imputed = scaler.inverse_transform(X_imputed)

        if np.isnan(np.sum(X_imputed)):
            raise WarningMessage("Result contains NaN. This is a bug.")

        if isinstance(X, np.ndarray):
            return X_imputed
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

        else:
            raise AssertionError("Invalid type. X must be either pd.DataFrame or np.ndarray.")
