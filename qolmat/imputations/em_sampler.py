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


class ImputeEM(_BaseImputer):  # type: ignore
    """
    Imputation of missing values using a multivariate Gaussian model through EM optimization and
    using a projected Ornstein-Uhlenbeck process.

    Parameters
    ----------
    method : str
        Method for imputation, choose among "sample" or "mle".
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
        n_iter_em: Optional[int] = 50,
        n_iter_ou: Optional[int] = 50,
        n_iter_mh: Optional[int] = 20,
        ampli: Optional[int] = 1,
        random_state: Optional[int] = 123,
        verbose: Optional[bool] = True,
        temporal: Optional[bool] = False,
        dt: Optional[float] = 2e-2,
        convergence_threshold: Optional[float] = 1e-4,
        stagnation_threshold: Optional[float] = 5e-3,
        stagnation_loglik: Optional[float] = 1e1,
    ) -> None:

        if strategy not in ["mle", "ou", "mh"]:
            raise Exception("strategy has to be 'mle' or 'ou' or 'mh'")

        self.strategy = strategy
        self.n_iter_em = n_iter_em
        self.n_iter_ou = n_iter_ou
        self.n_iter_mh = n_iter_mh
        self.ampli = ampli
        self.random_state = random_state
        self.verbose = verbose
        self.mask_outliers = None
        self.cov = None
        self.temporal = temporal
        self.dt = dt
        self.convergence_threshold = convergence_threshold
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

    def _em_mle(self, X: np.ndarray, observed: np.ndarray, missing: np.ndarray) -> np.ndarray:
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

            mu_tilde[i] = self.means[np.ix_(missing_i)] + sigma_MO @ np.linalg.inv(sigma_OO) @ (
                X[i, observed_i] - self.means[np.ix_(observed_i)]
            )
            sigma_MM_O = sigma_MM - sigma_MO @ np.linalg.inv(sigma_OO) @ sigma_OM
            sigma_tilde[i][np.ix_(missing_i, missing_i)] = sigma_MM_O
            X[i, missing_i] = mu_tilde[i]

        self.means = np.mean(X, axis=0)
        self.cov = np.cov(X.T, bias=1) + reduce(np.add, sigma_tilde.values()) / nr

        return X

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
        beta = self.cov.copy()
        beta[:] = scipy.linalg.inv(beta)
        gamma = np.diagonal(self.cov)
        X_stack = []
        for iter_ou in range(self.n_iter_ou):
            noise = self.ampli * rng.normal(0, 1, size=(n_samples, n_variables))
            X += gamma * ((self.means - X) @ beta) * dt + noise * np.sqrt(2 * gamma * dt)
            X[~mask_na] = X_init[~mask_na]
            if iter_ou > self.n_iter_ou - 50:
                X_stack.append(X)

        X_stack = np.vstack(X_stack)

        self.means = np.mean(X_stack, axis=0)
        self.cov = np.cov(X_stack.T, bias=True)
        eps = 1e-2
        self.cov -= eps * (self.cov - np.diag(self.cov.diagonal()))

        return X

    def _normal(self, X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
        numerator = np.exp(-0.5 * (X - mu) @ np.linalg.inv(cov) @ (X - mu).T)
        # denominator = np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(cov))
        return numerator  # / denominator

    def _tmultivariate(
        self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: float
    ) -> np.ndarray:
        p = len(X)
        # term1 = scipy.special.gamma((nu + p) / 2)
        # term2 = (
        #     scipy.special.gamma(nu / 2)
        #     * (nu * np.pi) ** (p / 2)
        #     * (np.linalg.det(sigma)) ** (1 / 2)
        # )
        term3 = (1 + (1.0 / nu) * (X - mu) @ np.linalg.inv(sigma) @ (X - mu).T) ** (
            -1.0 * (nu + p) / 2
        )
        return term3

    def _sample_mh(
        self,
        X: np.ndarray,
        mask_na: np.ndarray,
    ) -> float:

        _, n = X.shape
        X_ = X.copy()

        for _ in range(self.n_iter_mh):

            for i in range(X.shape[0]):
                if mask_na[i, :].sum() == 0:
                    continue
                move = X_[i, :] + np.random.normal(0, 1, size=n)
                # move = 0.5 * X_[i, :] + np.random.normal(scale=1, size=n)
                move[~mask_na[i, :]] = X[i, ~mask_na[i, :]]

                # curr_prob = self._normal(X_[i, :], mu=self.means, cov=self.cov)
                # move_prob = self._normal(move, mu=self.means, cov=self.cov)
                curr_prob = self._tmultivariate(X_[i, :], mu=self.means, sigma=self.cov, nu=n - 1)
                move_prob = self._tmultivariate(move, mu=self.means, sigma=self.cov, nu=n - 1)

                acceptance = min(move_prob / curr_prob, 1)
                if np.random.uniform(0, 1) < acceptance:
                    X_[i, :] = move

        self.means = np.mean(X_, axis=0)
        self.cov = np.cov(X_.T, bias=1)
        eps = 1e-2
        self.cov -= eps * (self.cov - np.diag(self.cov.diagonal()))

        return X_

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

        if self.strategy == "mle":
            mask = ~np.isnan(X_)
            one_to_nc = np.arange(1, nc + 1, step=1)
            observed = one_to_nc * mask - 1
            missing = one_to_nc * (~mask) - 1
        elif self.strategy in ["ou", "mh"]:
            mask_na = np.isnan(X)

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
        if np.abs(scipy.linalg.eigh(self.cov)[0].min()) > 1e20:
            raise WarningMessage("Large eigenvalues, imputation may be inflated.")

        if self.temporal:
            shift = 1
            X_past = X_transformed.copy()
            X_past = np.roll(X_past, shift, axis=0).astype(float)
            X_past[:shift, :] = np.nan

        list_X_transformed, list_means, list_covs = [], [], []
        for iter_em in range(self.n_iter_em):

            if self.temporal:
                shift = 1
                X_past = X_transformed.copy()
                X_past = np.roll(X_past, shift, axis=0).astype(float)
                X_past[:shift, :] = np.nan

            try:
                if self.strategy == "mle":
                    X_transformed = self._em_mle(X_transformed, observed, missing)
                elif self.strategy == "ou":
                    X_transformed = self._sample_ou(X_transformed, mask_na, rng, dt=2e-2)
                elif self.strategy == "mh":
                    X_transformed = self._sample_mh(X_transformed, mask_na)

                list_means.append(self.means)
                list_covs.append(self.cov)
                list_X_transformed.append(X_transformed)
                if (
                    self._check_convergence(list_X_transformed, list_means, list_covs, iter_em)
                    and self.verbose
                ):
                    print(f"EM converged after {iter_em} iterations.")
                    break

            except BaseException:
                raise WarningMessage("EM step failed.")

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
