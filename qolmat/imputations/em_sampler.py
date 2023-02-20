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
    index_imputed = np.isnan(X).any(axis=0)
    X_temp = X[:, index_imputed].copy()
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
    X_final[:, index_imputed] = X_temp

    return X_final

def invert_robust(M, epsilon=1e-2):
    # In case of inversibility problem, one can add a penalty term
    Meps = M - epsilon * (M - np.diag(M.diagonal()))
    if scipy.linalg.eigh(M)[0].min() < 0:
        print("---------------- FAILURE -------------")
        print(M.shape)
        print(M)
        print(scipy.linalg.eigh(M)[0].min())
        raise WarningMessage(
            f"Negative eigenvalue, some variables may be constant or colinear, "
            f"min value of {scipy.linalg.eigh(M)[0].min():.3g} found."
        )
    if np.abs(scipy.linalg.eigh(M)[0].min()) > 1e20:
        raise WarningMessage("Large eigenvalues, imputation may be inflated.")
    
    return scipy.linalg.inv(Meps)


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
        dt: Optional[float] = 2e-2,
        tolerance: Optional[float] = 1e-4,
        stagnation_threshold: Optional[float] = 5e-3,
        stagnation_loglik: Optional[float] = 2,
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
        self.dt = dt
        self.convergence_threshold = tolerance
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_loglik = stagnation_loglik

        self.dict_criteria_stop = {}

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
        self
    ) -> bool:
        return False

    def _maximize_likelihood(self, X: ArrayLike) -> ArrayLike:
        """
        Get the argmax of a posterior distribution.

        Parameters
        ----------
        X : ArrayLike
            Input DataFrame.

        Returns
        -------
        ArrayLike
            DataFrame with imputed values.
        """
        X_center = X - self.means[:, None]
        X_imputed = _gradient_conjugue(self.cov_inv, X_center)
        X_imputed = self.means[:, None] + X_imputed
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

        mask_na = np.isnan(X)

        # first imputation
        X_transformed = np.apply_along_axis(self._linear_interpolation, 1, X_)

        self.fit_distribution(X_transformed)
        self.dict_criteria_stop = {key: [] for key in self.dict_criteria_stop}

        for iter_em in range(self.max_iter_em):

            X_transformed = self._sample_ou(X_transformed, mask_na, rng)

            if (
                self._check_convergence()
            ):
                if self.verbose:
                    print(f"EM converged after {iter_em} iterations.")
                break
            
        if self.strategy == "mle":
            X_transformed = self._maximize_likelihood(X_)
        elif self.strategy == "ou":
            X_transformed = self._sample_ou(X_transformed, mask_na, rng)

        if np.all(np.isnan(X_transformed)):
            raise WarningMessage("Result contains NaN. This is a bug.")

        return X_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and impute input X array.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be imputed

        Returns
        -------
        ArrayLike
            Final array after EM sampling.
        """
        if not ((isinstance(df, np.ndarray)) or (isinstance(df, pd.DataFrame))):
            raise AssertionError("Invalid type. X must be either pd.DataFrame or np.ndarray.")

        if df.shape[1] < 2:
            raise AssertionError("Invalid dimensions: X must be of dimension (n,m) with m>1.")
        
        X = df.values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.T
        X = self.impute_em(X)
        X = X.T
        X = scaler.inverse_transform(X)

        if np.isnan(np.sum(X)):
            raise WarningMessage("Result contains NaN. This is a bug.")

        if isinstance(df, np.ndarray):
            return X
        elif isinstance(df, pd.DataFrame):
            return pd.DataFrame(X, index=df.index, columns=df.columns)

        else:
            raise AssertionError("Invalid type. X must be either pd.DataFrame or np.ndarray.")


class ImputeMultiNormalEM(ImputeEM):  # type: ignore
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
    dt : float
        Process integration time step, a large value increases the sample bias and can make
        the algorithm unstable, but compensates for a smaller n_iter_ou. By default, 2e-2.

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
        dt: Optional[float] = 2e-2,
        tolerance: Optional[float] = 1e-4,
        stagnation_threshold: Optional[float] = 5e-3,
        stagnation_loglik: Optional[float] = 1e1,
    ) -> None:

        super().__init__(strategy,max_iter_em,n_iter_ou,ampli,random_state,verbose,dt,stagnation_threshold,stagnation_loglik)
        self.tolerance = tolerance
        
        # self.list_logliks = []
        # self.list_means = []
        # self.list_covs = []
        self.dict_criteria_stop = {"logliks": [], "means": [], "covs": []}


    def fit_distribution(self, X):
        # first estimation of params
        self.means = np.mean(X, axis=1)
        self.cov = np.cov(X)
        
        self.cov_inv = invert_robust(self.cov, epsilon=1e-2)

    def _sample_ou(
        self,
        X: ArrayLike,
        mask_na: ArrayLike,
        rng: int,
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
        ampli : float
            Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

        Returns
        -------
        ArrayLike
            DataFrame after Ornstein-Uhlenbeck process.
        """
        n_samples, n_variables = X.shape

        X_init = X.copy()
        gamma = np.diagonal(self.cov)[:, None]
        list_means = []
        list_cov = []
        for iter_ou in range(self.n_iter_ou):
            noise = self.ampli * rng.normal(0, 1, size=(n_samples, n_variables))
            X_center = X - self.means[:, None]
            X = X - gamma * self.cov_inv @ X_center * self.dt + noise * np.sqrt(2 * gamma * self.dt)
            X[~mask_na] = X_init[~mask_na]
            if iter_ou > self.n_iter_ou - 50:
                list_means.append(np.mean(X, axis=1))
                list_cov.append(np.cov(X, bias=True))

        # MANOVA formula
        means_stack = np.stack(list_means, axis=1)
        self.means = np.mean(means_stack, axis=1)
        cov_stack = np.stack(list_cov, axis=2)
        self.cov = np.mean(cov_stack, axis=2) + np.cov(means_stack, bias=True)
        self.cov_inv = invert_robust(self.cov, epsilon=1e-2)

        self.loglik = scipy.stats.multivariate_normal.logpdf(X.T, self.means, self.cov).mean()

        self.dict_criteria_stop["means"].append(self.means)
        self.dict_criteria_stop["covs"].append(self.cov)
        self.dict_criteria_stop["logliks"].append(self.loglik)

        return X
    
    def _check_convergence(self) -> bool:
        """Check if the EM algorithm has converged. Three criteria:
        1) if the differences between the estimates of the parameters (mean and covariance) is
        less than a threshold (min_diff_reached - convergence_threshold).
        2) if the difference of the consecutive differences of the estimates is less than a
        threshold, i.e. stagnates over the last 5 interactions (min_diff_stable -
        stagnation_threshold).
        3) if the likelihood of the data no longer increases,
        i.e. stagnates over the last 5 iterations (max_loglik - stagnation_loglik).

        Returns
        -------
        bool
            True/False if the algorithm has converged
        """
        

        # self.list_means.append(self.means)
        # self.list_covs.append(self.cov)
        # self.list_logliks.append(self.loglik)

        list_means = self.dict_criteria_stop["means"]
        list_covs = self.dict_criteria_stop["covs"]
        list_logliks = self.dict_criteria_stop["logliks"]
        
        n_iter = len(list_means)

        min_diff_reached = (
            n_iter > 5
            and scipy.linalg.norm(list_means[-1] - list_means[-2], np.inf) < self.convergence_threshold
            and scipy.linalg.norm(list_covs[-1] - list_covs[-2], np.inf) < self.convergence_threshold
        )

        min_diff_stable = (
            n_iter > 10
            and min([scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_means[-6:], list_means[-5:])])
            < self.stagnation_threshold
            and min([scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_covs[-6:], list_covs[-5:])])
            < self.stagnation_threshold
        )

        if n_iter > 10:
            logliks = pd.Series(list_logliks[-6:])
            max_loglik = min(abs(logliks.diff())) < self.stagnation_loglik
        else:
            max_loglik = False

        return min_diff_reached or min_diff_stable or max_loglik


class ImputeVAR1EM(ImputeEM):  # type: ignore
    """
    Imputation of missing values using a vector autoregressive model through EM optimization and
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
        dt: Optional[float] = 2e-2,
        tolerance: Optional[float] = 1e-4,
        stagnation_threshold: Optional[float] = 5e-3,
        stagnation_loglik: Optional[float] = 1e1,
    ) -> None:

        super().__init__(strategy,max_iter_em,n_iter_ou,ampli,random_state,verbose,dt,stagnation_threshold,stagnation_loglik)
        self.tolerance = tolerance

    def fit_parameter_A(self, X):
        n_variables, n_samples = X.shape
        Xc = X - self.B[:, None]
        XX_lag = Xc[:, 1:] @ Xc[:, :-1].T
        XX = Xc @ Xc.T
        self.A = XX_lag @ invert_robust(XX, epsilon=1e-2)

    def fit_parameter_B(self, X):
        n_variables, n_samples = X.shape
        D = np.mean(X - self.A @ X, axis=1)
        self.B = scipy.linalg.inv(np.eye(n_variables) - self.A) @ D

    def fit_parameter_omega(self, X):
        n_variables, n_samples = X.shape
        Xc = X - self.B[:, None]
        Xc_lag = np.roll(Xc, 1)
        Z_back = Xc - self.A @ Xc_lag
        Z_back[:, 0] = 0
        self.omega = (Z_back @ Z_back.T) / n_samples
        self.omega_inv = invert_robust(self.omega, epsilon=1e-2)



    def fit_distribution(self, X):
        n_variables, n_samples = X.shape

        self.A = np.zeros((n_variables, n_variables))

        for n in range(5):
            self.fit_parameter_B(X)
            self.fit_parameter_A(X)
        self.fit_parameter_omega(X)

        # print("distribution fitted :")
        # print(self.A)
        # print(self.B)
        # print(self.omega)

    def gradient_X_centered_loglik(self, Xc):
        Xc_back = np.roll(Xc, 1, axis=1)
        Xc_back[:, 0] = 0
        Z_back = Xc - self.A @ Xc_back
        Xc_fore = np.roll(Xc, -1, axis=1)
        Xc_fore[:, -1] = 0
        Z_fore = Xc_fore - self.A @ Xc
        return - self.omega_inv @ Z_back + self.A.T @ self.omega_inv @ Z_fore

        

    def _sample_ou(
        self,
        X: ArrayLike,
        mask_na: ArrayLike,
        rng: int,
    ) -> ArrayLike:
        """
        Samples the Gaussian distribution under the constraint that not na values must remain
        unchanged, using a projected Ornstein-Uhlenbeck process.
        The sampled distribution tends to the target one in the limit dt -> 0 and
        n_iter_ou x dt -> infty.
        Called by `impute_sample_ts`.
        X = A X_t-1 + B + E where E ~ N(0, omega)

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
        ampli : float
            Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

        Returns
        -------
        ArrayLike
            DataFrame after Ornstein-Uhlenbeck process.
        """
        n_variables, n_samples = X.shape
        list_var = []

        X_init = X.copy()
        gamma = np.diagonal(self.omega)[:, None]
        Xc = X - self.B[:, None]
        Xc_init = X_init - self.B[:, None]
        for iter_ou in range(self.n_iter_ou):
            noise = self.ampli * rng.normal(0, 1, size=(n_variables, n_samples))
            
            # Xc_lag = np.roll(Xc, 1, axis=1)
            # Z_back = Xc - self.A @ Xc_lag
            # Z_fore = np.roll(Z_back, -1, axis=1)
            # Z_back[:, 0] = 0
            # Z_fore[:, -1] = 0
            # Xc = Xc - gamma * self.omega_inv @ Z_back * dt - gamma * self.A.T @ self.omega_inv @ Z_fore * dt + np.sqrt(2 * gamma[:, None] * dt) * noise
            grad_X = self.gradient_X_centered_loglik(Xc)
            
            # print("Xc")
            # print(Xc)
            # print(gamma)
            # print(grad_X)
            # print(noise)
            Xc = Xc + self.dt * gamma * grad_X + np.sqrt(2 * gamma * self.dt) * noise

            # from matplotlib import pyplot as plt
            # plt.figure(figsize=(24, 4))
            # plt.plot(Xc[0], label="before")
            # plt.plot((self.dt * gamma * grad_X)[0], label="grad")
            # plt.plot(Xc[0], label="after")
            # plt.ylim(-2, 2)
            # plt.legend()
            # plt.show()
            
            Xc[~mask_na] = Xc_init[~mask_na]
        X = Xc + self.B[:, None]

        self.fit_distribution(X)

        return X
    
    # def _sample_ou(
    #     self,
    #     X: ArrayLike,
    #     mask_na: ArrayLike,
    #     rng: int,
    #     dt: float = 2e-2,
    # ) -> ArrayLike:
    #     """
    #     Samples the Gaussian distribution under the constraint that not na values must remain
    #     unchanged, using a projected Ornstein-Uhlenbeck process.
    #     The sampled distribution tends to the target one in the limit dt -> 0 and
    #     n_iter_ou x dt -> infty.
    #     Called by `impute_sample_ts`.
    #     X = A X_t-1 + B + E where E ~ N(0, omega)

    #     Parameters
    #     ----------
    #     df : ArrayLike
    #         Inital dataframe to be imputed, which should have been already imputed using a simple
    #         method. This first imputation will be used as an initial guess.
    #     mask_na : ArrayLike
    #         Boolean dataframe indicating which coefficients should be resampled.
    #     rng : int
    #         Random number generator to be used (for reproducibility).
    #     n_iter_ou : int
    #         Number of iterations for the OU process, a large value decreases the sample bias
    #         but increases the computation time.
    #     dt : float
    #         Process integration time step, a large value increases the sample bias and can make
    #         the algorithm unstable, but compensates for a smaller n_iter_ou. By default, 2e-2.
    #     ampli : float
    #         Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

    #     Returns
    #     -------
    #     ArrayLike
    #         DataFrame after Ornstein-Uhlenbeck process.
    #     """
    #     n_variables, n_samples = X.shape

    #     X_init = X.copy()
    #     gamma = np.diagonal(self.omega)

    #     # list_X = []
    #     list_B = []
    #     list_XX_lag = []
    #     list_XX = []
    #     list_ZZ = []
    #     for iter_ou in range(self.n_iter_ou):
    #         noise = self.ampli * rng.normal(0, 1, size=(n_variables, n_samples))
    #         Xc = X - self.B[:, None]
    #         Xc_lag = np.roll(Xc, 1, axis=1)
    #         Z_back = Xc - self.A @ Xc_lag
    #         Z_back[:, 0] = 0
    #         Z_front = np.roll(Z_back, -1, axis=1)
    #         # X = X - self.omega_inv @ Z_back * dt - self.A.T @ self.omega_inv @ Z_front * dt + noise * np.sqrt(2 * dt)
    #         X = X - gamma * self.omega_inv @ Z_back * dt - gamma * self.A.T @ self.omega_inv @ Z_front * dt + np.sqrt(2 * gamma[:, None] * dt) * noise
    #         X[~mask_na] = X_init[~mask_na]
    #         if iter_ou > self.n_iter_ou - 50:
    #             X_lag = np.roll(X, 1, axis=1)
    #             B = np.mean(X - self.A @ X_lag, axis=1)

    #             Xc = X - self.B[:, None]
    #             Xc_lag = X_lag  - self.B[:, None]

    #             XX = X @ X.T
    #             XX_lag = Xc[:, 1:] @ X_lag[:, :-1].T
                
    #             Z_back = Xc - self.A @ Xc_lag
    #             Z_back[:, 0] = 0

    #             list_B.append(B) 
    #             list_XX.append(XX)
    #             list_XX_lag.append(XX_lag)
    #             list_ZZ.append(Z_back @ Z_back.T)

    #     B_stack = np.stack(list_B, axis=1)
    #     self.B = np.mean(B_stack, axis=1)

    #     XX_stack = np.stack(list_XX, axis=2)
    #     # XX_mean = np.mean(XX_stack, axis=2)
    #     XX_lag_stack = np.stack(list_XX_lag, axis=2)
    #     # XX_lag_mean = np.mean(XX_lag_stack, axis=2)

    #     # MANOVA formula
    #     cov_B = np.cov(B_stack, bias=True)
    #     cov_XX = np.mean(XX_stack, axis=2) + cov_B
    #     cov_XX_lag = np.mean(XX_lag_stack, axis=2) + cov_B
    #     self.A = cov_XX_lag @ invert_robust(cov_XX, epsilon=1e-2)
    #     ZZ_stack = np.stack(list_ZZ, axis=2)

    #     # Biased estimation of omega, ignoring intra-group covariances
    #     self.omega = np.mean(ZZ_stack, axis=2) # / n_variables
    #     self.omega_inv = invert_robust(self.omega, epsilon=1e-2)

    #     return X