from typing import Dict, List, Literal, Optional, Union
from warnings import WarningMessage

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from qolmat.utils import utils


def _gradient_conjugue(A: NDArray, X: NDArray, mask_na: NDArray) -> NDArray:
    """
    Minimize Tr(X.T AX) by imputing missing values.
    To this aim, we compute in parallel a gradient algorithm for each data.

    Parameters
    ----------
    A : NDArray
        Matrix defining the quadratic optimization problem
    X : NDArray
        Array containing the values to optimize
    mask_na : NDArray
        Boolean array indicating if a value of X is a variable of the optimization

    Returns
    -------
    NDArray
        Minimized array.
    """
    cols_imputed = mask_na.any(axis=0)
    X_temp = X[:, cols_imputed].copy()
    mask_na = mask_na[:, cols_imputed].copy()
    n_iter = mask_na.sum(axis=0).max()
    X_temp[mask_na] = 0
    b = -A @ X_temp
    b[~mask_na] = 0
    xn, pn, rn = np.zeros(X_temp.shape), b, b  # Initialisation
    for n in range(n_iter + 2):
        # if np.max(np.sum(rn**2)) < tol : # Condition de sortie " usuelle "
        #     X_temp[mask_isna] = xn[mask_isna]
        #     return X_temp.transpose()
        Apn = A @ pn
        Apn[~mask_na] = 0
        alphan = np.sum(rn**2, axis=0) / np.sum(pn * Apn, axis=0)
        alphan[np.isnan(alphan)] = 0  # we stop updating if convergence is reached for this date
        xn, rnp1 = xn + alphan * pn, rn - alphan * Apn
        betan = np.sum(rnp1**2, axis=0) / np.sum(rn**2, axis=0)
        betan[np.isnan(betan)] = 0  # we stop updating if convergence is reached for this date
        pn, rn = rnp1 + betan * pn, rnp1

    X_temp[mask_na] = xn[mask_na]
    X_final = X.copy()
    X_final[:, cols_imputed] = X_temp

    return X_final


class EM(BaseEstimator, TransformerMixin):
    """
    Generic class for missing values imputation through EM optimization and
    a projected Ornstein-Uhlenbeck process.

    Parameters
    ----------
    method : Literal["mle", "sample"]
        Method for imputation, choose among "mle" or "sample".
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
    >>> imputor = ImputeEM(method="mle")
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 3, 2],
    >>>                        [1, 2, 2, 1], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        method: Literal["mle", "sample"] = "sample",
        max_iter_em: int = 200,
        n_iter_ou: int = 50,
        ampli: float = 1,
        random_state: Union[None, int, np.random.RandomState] = None,
        dt: float = 2e-2,
        tolerance: float = 1e-4,
        stagnation_threshold: float = 5e-3,
        stagnation_loglik: float = 2,
        period: int = 1,
        verbose: bool = False,
    ):
        if method not in ["mle", "sample"]:
            raise ValueError(f"`method` must be 'mle' or 'sample', provided value is '{method}'")

        self.method = method
        self.max_iter_em = max_iter_em
        self.n_iter_ou = n_iter_ou
        self.ampli = ampli
        self.rng = sku.check_random_state(random_state)
        self.cov = np.array([[]])
        self.dt = dt
        self.tolerance = tolerance
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_loglik = stagnation_loglik
        self.scaler = StandardScaler()

        self.dict_criteria_stop: Dict[str, List] = {}
        self.period = period
        self.verbose = verbose

    def _convert_numpy(self, X: NDArray) -> NDArray:
        """
        Convert X pd.DataFrame to an array for internal calculations.

        Parameters
        ----------
        X : NDArray
            Input Array.

        Returns
        -------
        NDArray
            Return Array.
        """
        if not isinstance(X, np.ndarray):
            if (not isinstance(X, pd.DataFrame)) & (not isinstance(X, list)):
                raise ValueError("Input array is not a list, NDArray, nor pd.DataFrame.")
            X = X.to_numpy()
        return X

    def _check_convergence(self) -> bool:
        return False

    def fit(self, X: NDArray):
        """
        Fit the statistical distribution with the input X array.

        Parameters
        ----------
        X : NDArray
            Numpy array to be imputed
        """
        X = X.copy()
        self.hash_fit = hash(X.tobytes())
        if not isinstance(X, np.ndarray):
            raise AssertionError("Invalid type. X must be a NDArray.")

        X = utils.prepare_data(X, self.period)
        X = self.scaler.fit_transform(X.T).T

        mask_na = np.isnan(X)

        # first imputation
        X = utils.linear_interpolation(X)
        print("fit")
        print(X)
        print("fit_distribution")
        self.fit_distribution(X)
        print("...")

        for iter_em in range(self.max_iter_em):
            X = self._sample_ou(X, mask_na)

            if self._check_convergence():
                # print(f"EM converged after {iter_em} iterations.")
                break

        self.dict_criteria_stop = {key: [] for key in self.dict_criteria_stop}
        self.X_sample_last = X
        return self

    def transform(self, X: NDArray) -> NDArray:
        """
        Transform the input X array by imputing the missing values.

        Parameters
        ----------
        X : NDArray
            Numpy array to be imputed

        Returns
        -------
        NDArray
            Final array after EM sampling.
        """
        shape_original = X.shape
        if hash(X.tobytes()) == self.hash_fit:
            X = self.X_sample_last
        else:
            X = utils.prepare_data(X, self.period)
            X = self.scaler.transform(X.T).T
            X = utils.linear_interpolation(X)

        mask_na = np.isnan(X)

        if self.method == "mle":
            X_transformed = self._maximize_likelihood(X, mask_na, self.dt)
        elif self.method == "sample":
            X_transformed = self._sample_ou(X, mask_na)

        if np.all(np.isnan(X_transformed)):
            raise AssertionError("Result contains NaN. This is a bug.")

        X_transformed = self.scaler.inverse_transform(X_transformed.T).T
        X_transformed = utils.get_shape_original(X_transformed, shape_original)
        return X_transformed


class MultiNormalEM(EM):
    """
    Imputation of missing values using a multivariate Gaussian model through EM optimization and
    using a projected Ornstein-Uhlenbeck process.

    Parameters
    ----------
    method : Literal["mle", "sample"]
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
    dt : float
        Process integration time step, a large value increases the sample bias and can make
        the algorithm unstable, but compensates for a smaller n_iter_ou. By default, 2e-2.
    verbose: bool
        default `False`

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
    >>> imputor = ImputeEM(method="sample")
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 3, 2],
    >>>                        [1, 2, 2, 1], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputor.fit_transform(X)
    """

    def __init__(
        self,
        method: Literal["mle", "sample"] = "sample",
        max_iter_em: int = 200,
        n_iter_ou: int = 50,
        ampli: float = 1,
        random_state: Union[None, int, np.random.RandomState] = None,
        dt: float = 2e-2,
        tolerance: float = 1e-4,
        stagnation_threshold: float = 5e-3,
        stagnation_loglik: float = 2,
        period: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            method=method,
            max_iter_em=max_iter_em,
            n_iter_ou=n_iter_ou,
            ampli=ampli,
            random_state=random_state,
            dt=dt,
            tolerance=tolerance,
            stagnation_threshold=stagnation_threshold,
            stagnation_loglik=stagnation_loglik,
            period=period,
            verbose=verbose,
        )
        self.dict_criteria_stop = {"logliks": [], "means": [], "covs": []}

    def fit_distribution(self, X):
        self.means = np.mean(X, axis=1)
        n_rows, n_cols = X.shape
        if n_cols == 1:
            self.cov = np.eye(n_rows)
        else:
            self.cov = np.cov(X).reshape(n_rows, -1)
        self.cov_inv = np.linalg.pinv(self.cov, rcond=1e-2)

    def get_loglikelihood(self, X: NDArray) -> float:
        if np.all(np.isclose(self.cov, 0)):
            return 0
        else:
            return scipy.stats.multivariate_normal.logpdf(
                X.T, self.means, self.cov, allow_singular=True
            ).mean()

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray, dt: float = np.nan) -> NDArray:
        """
        Get the argmax of a posterior distribution.

        Parameters
        ----------
        X : NDArray
            Input DataFrame.

        Returns
        -------
        NDArray
            DataFrame with imputed values.
        """
        X_center = X - self.means[:, None]
        X_imputed = _gradient_conjugue(self.cov_inv, X_center, mask_na)
        X_imputed = self.means[:, None] + X_imputed
        return X_imputed

    def _sample_ou(
        self,
        X: NDArray,
        mask_na: NDArray,
    ) -> NDArray:
        """
        Samples the Gaussian distribution under the constraint that not na values must remain
        unchanged, using a projected Ornstein-Uhlenbeck process.
        The sampled distribution tends to the target one in the limit dt -> 0 and
        n_iter_ou x dt -> infty.
        Called by `impute_sample_ts`.

        Parameters
        ----------
        df : NDArray
            Inital dataframe to be imputed, which should have been already imputed using a simple
            method. This first imputation will be used as an initial guess.
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be resampled.
        n_iter_ou : int
            Number of iterations for the OU process, a large value decreases the sample bias
            but increases the computation time.
        ampli : float
            Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

        Returns
        -------
        NDArray
            DataFrame after Ornstein-Uhlenbeck process.
        """
        n_samples, n_variables = X.shape

        X_init = X.copy()
        gamma = np.diagonal(self.cov)[:, None]
        list_means = []
        list_cov = []
        for iter_ou in range(self.n_iter_ou):
            noise = self.ampli * self.rng.normal(0, 1, size=(n_samples, n_variables))
            X_center = X - self.means[:, None]
            X = (
                X
                - gamma * self.cov_inv @ X_center * self.dt
                + noise * np.sqrt(2 * gamma * self.dt)
            )
            X[~mask_na] = X_init[~mask_na]
            if iter_ou > self.n_iter_ou - 50:
                list_means.append(np.mean(X, axis=1))
                # reshaping for 1D input
                cov = np.cov(X, bias=True).reshape(len(X), -1)
                list_cov.append(cov)

        # MANOVA formula
        means_stack = np.stack(list_means, axis=1)
        self.means = np.mean(means_stack, axis=1)
        cov_stack = np.stack(list_cov, axis=2)
        self.cov = np.mean(cov_stack, axis=2) + np.cov(means_stack, bias=True)
        self.cov_inv = np.linalg.pinv(self.cov, rcond=1e-2)

        # Stop criteria
        self.loglik = self.get_loglikelihood(X)

        self.dict_criteria_stop["means"].append(self.means)
        self.dict_criteria_stop["covs"].append(self.cov)
        self.dict_criteria_stop["logliks"].append(self.loglik)

        return X

    def _check_convergence(self) -> bool:
        """
        Check if the EM algorithm has converged. Three criteria:
        1) if the differences between the estimates of the parameters (mean and covariance) is
        less than a threshold (min_diff_reached - tolerance).
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

        list_means = self.dict_criteria_stop["means"]
        list_covs = self.dict_criteria_stop["covs"]
        list_logliks = self.dict_criteria_stop["logliks"]

        n_iter = len(list_means)

        min_diff_reached = (
            n_iter > 5
            and scipy.linalg.norm(list_means[-1] - list_means[-2], np.inf) < self.tolerance
            and scipy.linalg.norm(list_covs[-1] - list_covs[-2], np.inf) < self.tolerance
        )

        min_diff_stable = (
            n_iter > 10
            and min(
                [
                    scipy.linalg.norm(t - s, np.inf)
                    for s, t in zip(list_means[-6:], list_means[-5:])
                ]
            )
            < self.stagnation_threshold
            and min(
                [scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_covs[-6:], list_covs[-5:])]
            )
            < self.stagnation_threshold
        )

        if n_iter > 10:
            logliks = pd.Series(list_logliks[-6:])
            max_loglik = min(abs(logliks.diff())) < self.stagnation_loglik
        else:
            max_loglik = False

        return min_diff_reached or min_diff_stable or max_loglik


class VAR1EM(EM):
    """
    Imputation of missing values using a vector autoregressive model through EM optimization and
    using a projected Ornstein-Uhlenbeck process.

    (X_n+1 - B) = A @ (X_n - B) + Omega @ G_n

    Parameters
    ----------
    method : Literal["mle", "sample"]
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
    dt : float
        Process integration time step, a large value increases the sample bias and can make
        the algorithm unstable, but compensates for a smaller n_iter_ou. By default, 2e-2.
    verbose: bool
        default `False`

    Attributes
    ----------
    X_intermediate : list
        List of pd.DataFrame giving the results of the EM process as function of the
        iteration number.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.em_sampler import VAR1EM
    >>> imputer = VAR1EM(method="sample")
    >>> X = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                        [np.nan, np.nan, 3, 2],
    >>>                        [1, 2, 2, 1], [2, 2, 2, 2]],
    >>>                        columns=["var1", "var2", "var3", "var4"])
    >>> imputer.fit_transform(X)
    """

    def __init__(
        self,
        method: Literal["mle", "sample"] = "sample",
        max_iter_em: int = 200,
        n_iter_ou: int = 50,
        ampli: float = 1,
        random_state: Union[None, int, np.random.RandomState] = None,
        dt: float = 2e-2,
        tolerance: float = 1e-4,
        stagnation_threshold: float = 5e-3,
        stagnation_loglik: float = 2,
        period: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            method=method,
            max_iter_em=max_iter_em,
            n_iter_ou=n_iter_ou,
            ampli=ampli,
            random_state=random_state,
            dt=dt,
            tolerance=tolerance,
            stagnation_threshold=stagnation_threshold,
            stagnation_loglik=stagnation_loglik,
            period=period,
            verbose=verbose,
        )

    def fit_parameter_A(self, X):
        n_variables, n_samples = X.shape
        Xc = X - self.B[:, None]
        XX_lag = Xc[:, 1:] @ Xc[:, :-1].T
        XX = Xc @ Xc.T
        XX_inv = np.linalg.pinv(XX, rcond=1e-2)
        self.A = XX_lag @ XX_inv

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
        self.omega_inv = np.linalg.pinv(self.omega, rcond=1e-2)

    def fit_distribution(self, X):
        n_variables, n_samples = X.shape

        self.A = np.zeros((n_variables, n_variables))

        for n in range(5):
            self.fit_parameter_B(X)
            self.fit_parameter_A(X)
        self.fit_parameter_omega(X)

    def gradient_X_centered_loglik(self, Xc):
        Xc_back = np.roll(Xc, 1, axis=1)
        Xc_back[:, 0] = 0
        Z_back = Xc - self.A @ Xc_back
        Xc_fore = np.roll(Xc, -1, axis=1)
        Xc_fore[:, -1] = 0
        Z_fore = Xc_fore - self.A @ Xc
        return -self.omega_inv @ Z_back + self.A.T @ self.omega_inv @ Z_fore

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray, dt=1e-2) -> NDArray:
        """
        Get the argmax of a posterior distribution.

        Parameters
        ----------
        X : NDArray
            Input numpy array.

        Returns
        -------
        NDArray
            DataFrame with imputed values.
        """
        Xc = X - self.B[:, None]
        for n_optim in range(1000):
            grad = self.gradient_X_centered_loglik(Xc)
            grad[~mask_na] = 0
            Xc += dt * grad
        return Xc + self.B[:, None]

    def _sample_ou(
        self,
        X: NDArray,
        mask_na: NDArray,
    ) -> NDArray:
        """
        Samples the Gaussian distribution under the constraint that not na values must remain
        unchanged, using a projected Ornstein-Uhlenbeck process.
        The sampled distribution tends to the target one in the limit dt -> 0 and
        n_iter_ou x dt -> infty.
        Called by `impute_sample_ts`.
        X = A X_t-1 + B + E where E ~ N(0, omega)

        Parameters
        ----------
        df : NDArray
            Inital dataframe to be imputed, which should have been already imputed using a simple
            method. This first imputation will be used as an initial guess.
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be resampled.
        n_iter_ou : int
            Number of iterations for the OU process, a large value decreases the sample bias
            but increases the computation time.
        ampli : float
            Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

        Returns
        -------
        NDArray
            DataFrame after Ornstein-Uhlenbeck process.
        """
        n_variables, n_samples = X.shape

        X_init = X.copy()
        gamma = np.diagonal(self.omega)[:, None]
        Xc = X - self.B[:, None]
        Xc_init = X_init - self.B[:, None]
        for iter_ou in range(self.n_iter_ou):
            noise = self.ampli * self.rng.normal(0, 1, size=(n_variables, n_samples))
            grad_X = self.gradient_X_centered_loglik(Xc)

            Xc = Xc + self.dt * gamma * grad_X + np.sqrt(2 * gamma * self.dt) * noise

            Xc[~mask_na] = Xc_init[~mask_na]
        X = Xc + self.B[:, None]

        self.fit_distribution(X)

        return X
