from abc import abstractmethod
from typing import Dict, List, Literal, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from qolmat.utils import utils


def _conjugate_gradient(A: NDArray, X: NDArray, mask: NDArray) -> NDArray:
    """
    Minimize Tr(X.T AX) wrt X where X is constrained to the initial value outside the given mask
    To this aim, we compute in parallel a gradient algorithm for each row.

    Parameters
    ----------
    A : NDArray
        Symmetrical matrix defining the quadratic optimization problem
    X : NDArray
        Array containing the values to optimize
    mask : NDArray
        Boolean array indicating if a value of X is a variable of the optimization

    Returns
    -------
    NDArray
        Minimized array.
    """
    rows_imputed = mask.any(axis=1)
    X_temp = X[rows_imputed, :].copy()
    mask = mask[rows_imputed, :].copy()
    n_iter = mask.sum(axis=1).max()
    X_temp[mask] = 0
    b = -X_temp @ A
    b[~mask] = 0
    xn, pn, rn = np.zeros(X_temp.shape), b, b  # Initialisation
    for n in range(n_iter + 2):
        # if np.max(np.sum(rn**2)) < tol : # Condition de sortie " usuelle "
        #     X_temp[mask_isna] = xn[mask_isna]
        #     return X_temp.transpose()
        Apn = pn @ A
        Apn[~mask] = 0
        alphan = np.sum(rn**2, axis=1) / np.sum(pn * Apn, axis=1)
        alphan[np.isnan(alphan)] = 0  # we stop updating if convergence is reached for this date
        xn, rnp1 = xn + pn * alphan[:, None], rn - Apn * alphan[:, None]
        betan = np.sum(rnp1**2, axis=1) / np.sum(rn**2, axis=1)
        betan[np.isnan(betan)] = 0  # we stop updating if convergence is reached for this date
        pn, rn = rnp1 + pn * betan[:, None], rnp1

    X_temp[mask] = xn[mask]
    X_final = X.copy()
    X_final[rows_imputed, :] = X_temp

    return X_final


def fit_var_model(data: NDArray, p: int, criterion: str = "aic") -> VARResultsWrapper:
    model = VAR(data)
    result = model.fit(maxlags=p, ic=criterion)
    return result


def get_lag_p(X: NDArray, max_lag_order: int = 10, criterion: str = "aic") -> int:
    if criterion not in ["aic", "bic"]:
        raise AssertionError("Invalid criterion. `criterion` must be `aic`or `bic`.")

    best_p = 1
    best_criteria_value = float("inf")
    for p in range(1, max_lag_order + 1):
        model_result = fit_var_model(X, p, criterion=criterion)
        if criterion == "aic":
            criteria_value = model_result.aic
        else:
            criteria_value = model_result.bic

        if criteria_value < best_criteria_value:
            best_p = p
            best_criteria_value = criteria_value

    return best_p


class EM(BaseEstimator, TransformerMixin):
    """
    Generic abstract class for missing values imputation through EM optimization and
    a projected MCMC sampling process.

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
    """

    def __init__(
        self,
        method: Literal["mle", "sample"] = "sample",
        max_iter_em: int = 500,
        n_iter_ou: int = 50,
        n_samples: int = 10,
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
        self.n_samples = n_samples

    def _check_convergence(self) -> bool:
        return False

    def fit_parameters(self, X: NDArray):
        self.reset_learned_parameters()
        self.update_parameters(X)
        self.combine_parameters()

    @abstractmethod
    def gradient_X_centered_loglik(
        self,
        X: NDArray,
        mask_na: NDArray,
    ) -> NDArray:
        return np.empty

    def get_gamma(self) -> NDArray:
        n_rows, n_cols = self.shape_original
        return np.ones((1, n_cols))

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray, dt=1e-2) -> NDArray:
        """Get the argmax of a posterior distribution.

        Parameters∑
        ----------
        X : NDArray
            Input numpy array.

        Returns
        -------
        NDArray
            DataFrame with imputed values.
        """
        for _ in range(1000):
            grad = self.gradient_X_loglik(X)
            grad[~mask_na] = 0
            X += dt * grad
        return X

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

        Returns
        -------
        NDArray
            Sampled data matrix
        """
        n_variables, n_samples = X.shape
        self.reset_learned_parameters()
        X_init = X.copy()
        gamma = self.get_gamma()
        for _ in range(self.n_iter_ou):
            noise = self.ampli * self.rng.normal(0, 1, size=(n_variables, n_samples))
            grad_X = self.gradient_X_loglik(X)
            X -= self.dt * gamma * grad_X + np.sqrt(2 * gamma * self.dt) * noise
            X[~mask_na] = X_init[~mask_na]
            self.update_parameters(X)

        return X

    def fit(self, X: NDArray) -> Self:
        """
        Fit the statistical distribution with the input X array.

        Parameters
        ----------
        X : NDArray
            Numpy array to be imputed
        """
        X = X.copy()
        self.shape_original = X.shape

        self.hash_fit = hash(X.tobytes())
        if not isinstance(X, np.ndarray):
            raise AssertionError("Invalid type. X must be a NDArray.")

        X = utils.prepare_data(X, self.period)
        X = self.scaler.fit_transform(X)

        if hasattr(self, "p"):
            if self.p is None:  # type: ignore # noqa
                self.p = get_lag_p(utils.linear_interpolation(X))
            self.n_features = X.shape[1]
            # X = utils.create_lag_matrix(X, self.p)
            Z, Y = utils.create_lag_matrices(X, self.p)

        mask_na = np.isnan(X)

        # first imputation
        X = utils.linear_interpolation(X)
        self.fit_parameters(X)

        for iter_em in range(self.max_iter_em):
            X = self._sample_ou(X, mask_na)
            self.combine_parameters()

            # Stop criteria
            self.loglik = self.get_loglikelihood(X)
            self.update_criteria_stop()
            if self._check_convergence():
                print(f"EM converged after {iter_em} iterations.")
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
        # shape_original = X.shape
        if hash(X.tobytes()) == self.hash_fit:
            X = self.X_sample_last
        else:
            X = utils.prepare_data(X, self.period)
            X = self.scaler.transform(X)
            X = utils.linear_interpolation(X)
            if hasattr(self, "p"):
                if self.p is None:
                    self.p = get_lag_p(utils.linear_interpolation(X))
                X = self.create_lag_matrix_X(X)

        mask_na = np.isnan(X)

        if self.method == "mle":
            X_transformed = self._maximize_likelihood(X, mask_na, self.dt)
        elif self.method == "sample":
            X_transformed = self._sample_ou(X, mask_na)

        if np.all(np.isnan(X_transformed)):
            raise AssertionError("Result contains NaN. This is a bug.")

        # if hasattr(self, "p"):
        #     X_transformed = self.get_X(X_transformed)
        #     self.get_parameters_ABomega()
        #     self.dict_criteria_stop["As"] = self.A
        #     self.dict_criteria_stop["Bs"] = self.B
        #     self.dict_criteria_stop["omegas"] = self.omega

        X_transformed = self.scaler.inverse_transform(X_transformed)
        # X_transformed = utils.get_shape_original(X_transformed, self.shape_original)

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

    def get_gamma(self) -> NDArray:
        gamma = np.diagonal(self.cov).reshape(1, -1)
        return gamma

    def update_criteria_stop(self):
        self.dict_criteria_stop["means"].append(self.means)
        self.dict_criteria_stop["covs"].append(self.cov)
        self.dict_criteria_stop["logliks"].append(self.loglik)

    def reset_learned_parameters(self):
        self.list_means = []
        self.list_cov = []

    def update_parameters(self, X):
        n_rows, n_cols = X.shape
        means = np.mean(X, axis=0)
        self.list_means.append(means)
        # reshaping for 1D input
        cov = np.cov(X, bias=True, rowvar=False).reshape(n_cols, -1)
        self.list_cov.append(cov)

    def combine_parameters(self):
        list_means = self.list_means[-self.n_samples :]
        list_cov = self.list_cov[-self.n_samples :]
        # self.means = np.mean(X, axis=0)
        # n_rows, n_cols = X.shape
        # if n_cols == 1:
        #     self.cov = np.eye(n_cols)
        # else:
        #     self.cov = np.cov(X, rowvar=False).reshape(n_cols, -1)
        # self.cov_inv = np.linalg.pinv(self.cov, rcond=1e-2)

        # MANOVA formula
        means_stack = np.stack(list_means, axis=0)
        self.means = np.mean(means_stack, axis=0)
        cov_stack = np.stack(list_cov, axis=2)
        self.cov = np.mean(cov_stack, axis=2) + np.cov(means_stack, bias=True, rowvar=False)
        self.cov_inv = np.linalg.pinv(self.cov, rcond=1e-2)

    def get_loglikelihood(self, X: NDArray) -> float:
        if np.all(np.isclose(self.cov, 0)):
            return 0
        else:
            return scipy.stats.multivariate_normal.logpdf(
                X, self.means, self.cov, allow_singular=True
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
        X_imputed = _conjugate_gradient(self.cov_inv, X_center, mask_na)
        X_imputed = self.means[:, None] + X_imputed
        return X_imputed

    def gradient_X_loglik(self, X: NDArray) -> NDArray:
        """
        Calculates the gradient of a centered
        log-likelihood function using a given matrix.
        (X_t+1 - A * X_t).T omega_inv (X_t+1 - A * X_t)

        Parameters
        ----------
        Xc : NDArray
            Xc is a numpy array representing the centered input data.

        Returns
        -------
        NDArray
            The gradient of the centered log-likelihood with respect to the input variable `Xc`.
        """
        grad_X = (X - self.means) @ self.cov_inv
        return grad_X

    # def _sample_ou(
    #     self,
    #     X: NDArray,
    #     mask_na: NDArray,
    # ) -> NDArray:
    #     """
    #     Samples the Gaussian distribution under the constraint that not na values must remain
    #     unchanged, using a projected Ornstein-Uhlenbeck process.
    #     The sampled distribution tends to the target one in the limit dt -> 0 and
    #     n_iter_ou x dt -> infty.
    #     Called by `impute_sample_ts`.

    #     Parameters
    #     ----------
    #     df : NDArray
    #         Inital dataframe to be imputed, which should have been already imputed using a simple
    #         method. This first imputation will be used as an initial guess.
    #     mask_na : NDArray
    #         Boolean dataframe indicating which coefficients should be resampled.
    #     n_iter_ou : int
    #         Number of iterations for the OU process, a large value decreases the sample bias
    #         but increases the computation time.
    #     ampli : float
    #         Amplification of the noise, if less than 1 the variance is reduced. By default, 1.

    #     Returns
    #     -------
    #     NDArray
    #         DataFrame after Ornstein-Uhlenbeck process.
    #     """
    #     n_samples, n_variables = X.shape
    #     print("_sample_ou")
    #     print(X.shape)

    #     X_init = X.copy()
    #     gamma = np.diagonal(self.cov)
    #     list_means = []
    #     list_cov = []
    #     nb_samples = 50
    #     for iter_ou in range(self.n_iter_ou):
    #         noise = self.ampli * self.rng.normal(0, 1, size=(n_samples, n_variables))
    #         X_center = X - self.means
    #         X += -X_center @ self.cov_inv * gamma * self.dt
    # + noise * np.sqrt(2 * gamma * self.dt)
    #         X[~mask_na] = X_init[~mask_na]
    #         if iter_ou > self.n_iter_ou - nb_samples:
    #             list_means.append(np.mean(X, axis=0))
    #             # reshaping for 1D input
    #             cov = np.cov(X, bias=True, rowvar=False).reshape(n_variables, -1)
    #             list_cov.append(cov)

    #     # MANOVA formula
    #     means_stack = np.stack(list_means, axis=0)
    #     self.means = np.mean(means_stack, axis=0)
    #     cov_stack = np.stack(list_cov, axis=2)
    #     self.cov = np.mean(cov_stack, axis=2) + np.cov(means_stack, bias=True, rowvar=False)
    #     self.cov_inv = np.linalg.pinv(self.cov, rcond=1e-2)

    #     # Stop criteria
    #     self.loglik = self.get_loglikelihood(X)

    #     self.dict_criteria_stop["means"].append(self.means)
    #     self.dict_criteria_stop["covs"].append(self.cov)
    #     self.dict_criteria_stop["logliks"].append(self.loglik)

    #     return X

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
            max_loglik = (min(abs(logliks.diff(1)[1:])) < self.stagnation_loglik) or (
                min(abs(logliks.diff(2)[2:])) < self.stagnation_loglik
            )
        else:
            max_loglik = False

        return min_diff_reached or min_diff_stable or max_loglik


class VARpEM(EM):
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
        p: Union[None, int] = None,
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
        self.dict_criteria_stop = {"logliks": [], "As": [], "Bs": [], "omegas": []}
        self.p = p  # type: ignore #noqa

    def get_X(self, X: NDArray) -> NDArray:
        return X[:, : self.n_features]

    def update_criteria_stop(self):
        self.dict_criteria_stop["As"].append(self.A)
        self.dict_criteria_stop["Bs"].append(self.B)
        self.dict_criteria_stop["omegas"].append(self.omega)
        self.dict_criteria_stop["logliks"].append(self.loglik)

    def get_parameters_ABomega(self) -> None:
        estimated_A = []
        for i in range(self.p):
            estimated_A.append(self.A[: self.n_features, (i * self.n_features) : (i * self.n_features) + self.n_features])  # type: ignore # noqa
        estimated_B = self.B[: self.n_features]  # type: ignore # noqa
        estimated_omega = self.omega[: self.n_features, : self.n_features]  # type: ignore # noqa

        self.list_A = estimated_A
        self.B = estimated_B
        self.omega = estimated_omega
        self.omega_inv = np.linalg.pinv(self.omega, rcond=1e-2)

    def fit_parameter_A(self, X: NDArray) -> None:
        """
        Calculates the parameter `A` using the input `X` and
        the previously calculated parameter `B`.

        Parameters
        ----------
        X : NDArray
            Input data.
        """
        Xc = X - self.B[:, None]  # type: ignore #noqa
        XX_lag = Xc[:, 1:] @ Xc[:, :-1].T
        XX = Xc[:, :-1] @ Xc[:, :-1].T
        XX_inv = np.linalg.pinv(XX, rcond=1e-2)
        self.A = XX_lag @ XX_inv

    def fit_parameter_B(self, X: NDArray) -> None:
        """
        Calculates the value of parameter `B` based on
        the input matrix `X` and the existing parameter `A`.

        Parameters
        ----------
        X : NDArray
            Input data
        """
        n_variables, _ = X.shape
        D = np.mean(X - self.A @ X, axis=1)
        self.B = scipy.linalg.inv(np.eye(n_variables) - self.A) @ D

    def fit_parameter_omega(self, X: NDArray) -> None:
        """
        Calculates the covariance matrix `omega` and
        its inverse `omega_inv` based on the input matrix `X`.

        Parameters
        ----------
        X : NDArray
            X is a numpy array representing the input data. It has shape (m, n),
            where m is the number of features and n is the number of samples.
        """
        _, n_samples = X.shape
        Xc = X - self.B[:, None]
        Xc_lag = np.roll(Xc, 1)
        Z_back = Xc - self.A @ Xc_lag
        Z_back[:, 0] = 0
        self.omega = (Z_back @ Z_back.T) / n_samples
        self.omega_inv = np.linalg.pinv(self.omega, rcond=1e-2)

    # def fit_distribution(self, X: NDArray) -> None:
    #     """
    #     Fits the parameters `A`, `B`, and `omega` of a VAR(p) process
    #     using the input data `X`.

    #     Parameters
    #     ----------
    #     X : NDArray
    #         Input data for which the distribution is being fitted.
    #     """
    #     n_variables, _ = X.shape

    #     self.A = np.zeros((n_variables, n_variables))  # type: ignore # noqa
    #     for _ in range(5):
    #         self.fit_parameter_B(X)
    #         self.fit_parameter_A(X)
    #     self.fit_parameter_omega(X)

    def update_parameters(self, X: NDArray) -> None:
        """
        Fits the parameters `beta` and `Sigma` of a VAR(p) process
        using the input data `X`.

        Parameters
        ----------
        X : NDArray
            Input data for which the distribution is being fitted.
        """
        n_rows, n_cols = X.shape
        Z, Y = utils.create_lag_matrices(X, self.p)
        self.list_ZZ.append(Z @ Z.T)

    def gradient_X_loglik(self, Xc: NDArray) -> NDArray:
        """
        Calculates the gradient of a centered
        log-likelihood function using a given matrix.
        (X_t+1 - A * X_t).T omega_inv (X_t+1 - A * X_t)

        Parameters
        ----------
        Xc : NDArray
            Xc is a numpy array representing the centered input data.

        Returns
        -------
        NDArray
            The gradient of the centered log-likelihood with respect to the input variable `Xc`.
        """
        Xc_back = np.roll(Xc, 1, axis=1)
        Xc_back[:, 0] = 0
        Z_back = Xc - self.A @ Xc_back
        Xc_fore = np.roll(Xc, -1, axis=1)
        Xc_fore[:, -1] = 0
        Z_fore = Xc_fore - self.A @ Xc
        return -self.omega_inv @ Z_back + self.A.T @ self.omega_inv @ Z_fore

    def get_loglikelihood(self, X: NDArray) -> float:
        p, n = X.shape
        sign, logdet = np.linalg.slogdet(self.omega)
        Xc = X - self.B[:, None]
        Xc_back = np.roll(Xc, 1, axis=1)
        Xc_back[:, 0] = 0
        return (
            -n * p / 2 * np.log(2 * np.pi)
            - n / 2 * sign * logdet
            - 1
            / 2
            * np.trace((Xc - self.A @ Xc_back).T @ self.omega_inv @ (Xc - self.A @ Xc_back))
        )

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

        list_As = self.dict_criteria_stop["As"]
        list_Bs = self.dict_criteria_stop["Bs"]
        list_omegas = self.dict_criteria_stop["omegas"]
        list_logliks = self.dict_criteria_stop["logliks"]

        n_iter = len(list_As)

        min_diff_reached = (
            n_iter > 5
            and scipy.linalg.norm(list_As[-1] - list_As[-2], np.inf) < self.tolerance
            and scipy.linalg.norm(list_Bs[-1] - list_Bs[-2], np.inf) < self.tolerance
            and scipy.linalg.norm(list_omegas[-1] - list_omegas[-2], np.inf) < self.tolerance
        )

        min_diff_stable = (
            n_iter > 10
            and min([scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_As[-6:], list_As[-5:])])
            < self.stagnation_threshold
            and min([scipy.linalg.norm(t - s, np.inf) for s, t in zip(list_Bs[-6:], list_Bs[-5:])])
            < self.stagnation_threshold
            and min(
                [
                    scipy.linalg.norm(t - s, np.inf)
                    for s, t in zip(list_omegas[-6:], list_omegas[-5:])
                ]
            )
            < self.stagnation_threshold
        )

        if n_iter > 10:
            logliks = pd.Series(list_logliks[-6:])
            max_loglik = (min(abs(logliks.diff(1)[1:])) < self.stagnation_loglik) or (
                min(abs(logliks.diff(2)[2:])) < self.stagnation_loglik
            )
        else:
            max_loglik = False

        return min_diff_reached or min_diff_stable or max_loglik
