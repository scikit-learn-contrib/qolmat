from abc import abstractmethod
from typing import Dict, List, Literal, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as spl
from scipy import optimize as spo
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self

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


def min_diff_Linf(list_params: List[NDArray], n_steps: int, order: int = 1) -> float:
    """Computes the maximal L infinity norm between the `n_steps` last elements spaced by order.
    Used to compute the stop criterion.

    Parameters
    ----------
    list_params : List[NDArray]
        List of statistics from the samples
    n_steps : int
        Number of steps to take into account
    order : int, optional
        Space between compared statistics, by default 1

    Returns
    -------
    float
        Minimal norm of differences
    """
    params = np.stack(list_params[-n_steps - order : -order])
    params_shift = np.stack(list_params[-n_steps:])
    min_diff = np.max(np.abs(params - params_shift))
    return min_diff


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
    n_samples : int, optional
        Number of data samples used to estimate the parameters of the distribution. Default, 10
    ampli : float, optional
        Whether to sample the posterior (1)
        or to maximise likelihood (0), by default 1.
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility.
    dt : float, optional
        Process integration time step, a large value increases the sample bias and can make
        the algorithm unstable, but compensates for a smaller n_iter_ou. By default, 2e-2.
    tolerance : float, optional
        Threshold below which a L infinity norm difference indicates the convergence of the
        parameters
    stagnation_threshold : float, optional
        Threshold below which a stagnation of the L infinity norm difference indicates the
        convergence of the parameters
    stagnation_loglik : float, optional
        Threshold below which an absolute difference of the log likelihood indicates the
        convergence of the parameters
    period : int, optional
        Integer used to fold the temporal data periodically
    verbose : bool, optional
        Verbosity level, if False the warnings are silenced
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

        self.dict_criteria_stop: Dict[str, List] = {}
        self.period = period
        self.verbose = verbose
        self.n_samples = n_samples

    def _check_convergence(self) -> bool:
        return False

    @abstractmethod
    def reset_learned_parameters(self):
        pass

    @abstractmethod
    def update_parameters(self, X: NDArray):
        pass

    @abstractmethod
    def combine_parameters(self):
        pass

    def fit_parameters(self, X: NDArray):
        self.reset_learned_parameters()
        self.update_parameters(X)
        self.combine_parameters()

    def update_criteria_stop(self, X: NDArray):
        self.loglik = self.get_loglikelihood(X)

    @abstractmethod
    def get_loglikelihood(self, X: NDArray) -> float:
        return 0

    @abstractmethod
    def gradient_X_loglik(
        self,
        X: NDArray,
    ) -> NDArray:
        return np.empty  # type: ignore #noqa

    def get_gamma(self) -> NDArray:
        n_rows, n_cols = self.shape_original
        return np.ones((1, n_cols))

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray) -> NDArray:
        """Get the argmax of a posterior distribution using the BFGS algorithm.

        Parameters
        ----------
        X : NDArray
            Input numpy array.
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be resampled, and are therefore
            the variables of the optimization

        Returns
        -------
        NDArray
            DataFrame with imputed values.
        """

        def fun_obj(x):
            x_mat = X.copy()
            x_mat[mask_na] = x
            return self.get_loglikelihood(x_mat)

        def fun_jac(x):
            x_mat = X.copy()
            x_mat[mask_na] = x
            grad_x = self.gradient_X_loglik(x_mat)
            grad_x[~mask_na] = 0
            return grad_x

        res = spo.minimize(fun_obj, X[mask_na], jac=fun_jac)

        # for _ in range(1000):
        #     grad = self.gradient_X_loglik(X)
        #     grad[~mask_na] = 0
        #     X += dt * grad
        x = res.x
        X_sol = X.copy()
        X_sol[mask_na] = x
        return X_sol

    def _sample_ou(
        self,
        X: NDArray,
        mask_na: NDArray,
        estimate_params: bool = True,
    ) -> NDArray:
        """
        Samples the Gaussian distribution under the constraint that not na values must remain
        unchanged, using a projected Ornstein-Uhlenbeck process.
        The sampled distribution tends to the target distribution in the limit dt -> 0 and
        n_iter_ou x dt -> infty.

        Parameters
        ----------
        df : NDArray
            Inital dataframe to be imputed, which should have been already imputed using a simple
            method. This first imputation will be used as an initial guess.
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be resampled.
        estimate_params : bool
            Indicates if the parameters of the distribution should be estimated while the data are
            sampled.

        Returns
        -------
        NDArray
            Sampled data matrix
        """
        X_copy = X.copy()
        n_variables, n_samples = X_copy.shape
        if estimate_params:
            self.reset_learned_parameters()
        X_init = X.copy()
        gamma = self.get_gamma()
        sqrt_gamma = np.real(spl.sqrtm(gamma))
        for i in range(self.n_iter_ou):
            noise = self.ampli * self.rng.normal(0, 1, size=(n_variables, n_samples))
            grad_X = self.gradient_X_loglik(X_copy)
            X_copy += self.dt * grad_X @ gamma + np.sqrt(2 * self.dt) * noise @ sqrt_gamma
            X_copy[~mask_na] = X_init[~mask_na]
            if estimate_params:
                self.update_parameters(X_copy)

        return X_copy

    def fit_X(self, X: NDArray) -> None:
        mask_na = np.isnan(X)

        # first imputation
        X = utils.linear_interpolation(X)
        self.fit_parameters(X)

        if not np.any(mask_na):
            self.X = X

        for iter_em in range(self.max_iter_em):
            X = self._sample_ou(X, mask_na)
            self.combine_parameters()

            # Stop criteria
            self.update_criteria_stop(X)
            if self._check_convergence():
                print(f"EM converged after {iter_em} iterations.")
                break

        self.dict_criteria_stop = {key: [] for key in self.dict_criteria_stop}
        self.X = X

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

        if hasattr(self, "p_to_fit") and self.p_to_fit:
            aics: List[float] = []
            for p in range(self.max_lagp + 1):
                self.p = p
                self.fit_X(X)
                n1, n2 = self.X.shape
                det = np.linalg.det(self.S)
                if abs(det) < 1e-12:
                    aic = -np.inf
                else:
                    aic = np.log(det) + 2 * p * (n2**2) / n1
                if len(aics) > 0 and aic > aics[-1]:
                    break
                aics.append(aic)
                if aic == -np.inf:
                    break
            self.p = int(np.argmin(aics))
            self.fit_X(X)

        else:
            self.fit_X(X)

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
        mask_na = np.isnan(X)

        # shape_original = X.shape
        if hash(X.tobytes()) == self.hash_fit:
            X = self.X
        else:
            X = utils.prepare_data(X, self.period)
            X = utils.linear_interpolation(X)

        if self.method == "mle":
            X_transformed = self._maximize_likelihood(X, mask_na)
        elif self.method == "sample":
            X_transformed = self._sample_ou(X, mask_na, estimate_params=False)

        if np.all(np.isnan(X_transformed)):
            raise AssertionError("Result contains NaN. This is a bug.")

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
    tolerance : float, optional
        Threshold below which a L infinity norm difference indicates the convergence of the
        parameters
    stagnation_threshold : float, optional
        Threshold below which a L infinity norm difference indicates the convergence of the
        parameters
    stagnation_loglik : float, optional
        Threshold below which an absolute difference of the log likelihood indicates the
        convergence of the parameters
    period : int, optional
        Integer used to fold the temporal data periodically
    verbose : bool, optional
        Verbosity level, if False the warnings are silenced
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

    def get_loglikelihood(self, X: NDArray) -> float:
        """
        Value of the log-likelihood up to a constant for the provided X, using the attributes
        `means` and `cov_inv` for the multivariate normal distribution.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column

        Returns
        -------
        float
            Computed value
        """
        Xc = X - self.means
        return -((Xc @ self.cov_inv) * Xc).sum().sum() / 2

    def gradient_X_loglik(self, X: NDArray) -> NDArray:
        """
        Gradient of the log-likelihood for the provided X, using the attributes
        `means` and `cov_inv` for the multivariate normal distribution.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column

        Returns
        -------
        NDArray
            The gradient of the log-likelihood with respect to the input variable `X`.
        """
        grad_X = -(X - self.means) @ self.cov_inv
        return grad_X

    def get_gamma(self) -> NDArray:
        """
        Normalisation matrix used to stabilize the sampling process

        Returns
        -------
        NDArray
            Gamma matrix
        """
        # gamma = np.diag(np.diagonal(self.cov))
        gamma = self.cov
        # gamma = np.eye(len(self.cov))
        return gamma

    def update_criteria_stop(self, X: NDArray):
        """
        Updates the variables which will be used to compute the stop critera

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column
        """
        self.loglik = self.get_loglikelihood(X)
        self.dict_criteria_stop["means"].append(self.means)
        self.dict_criteria_stop["covs"].append(self.cov)
        self.dict_criteria_stop["logliks"].append(self.loglik)

    def reset_learned_parameters(self):
        """
        Resets all lists of estimated parameters before starting a new estimation.
        """
        self.list_means = []
        self.list_cov = []

    def update_parameters(self, X):
        """
        Retains statistics relative to the current sample, in prevision of combining them.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column
        """
        n_rows, n_cols = X.shape
        means = np.mean(X, axis=0)
        self.list_means.append(means)
        # reshaping for 1D input
        if n_rows == 1:
            cov = np.zeros((n_cols, n_cols))
        else:
            cov = np.cov(X, bias=True, rowvar=False).reshape(n_cols, -1)
        self.list_cov.append(cov)

    def combine_parameters(self):
        """
        Combine all statistics computed for each sample in the update step, using the MANOVA
        formula.
        """
        list_means = self.list_means[-self.n_samples :]
        list_cov = self.list_cov[-self.n_samples :]

        # MANOVA formula
        means_stack = np.stack(list_means)
        self.means = np.mean(means_stack, axis=0)
        cov_stack = np.stack(list_cov)
        cov_intragroup = np.mean(cov_stack, axis=0)
        if len(list_means) == 1:
            cov_intergroup = np.zeros(cov_intragroup.shape)
        else:
            cov_intergroup = np.cov(means_stack, bias=True, rowvar=False)
        self.cov = cov_intragroup + cov_intergroup
        self.cov_inv = np.linalg.pinv(self.cov)

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray) -> NDArray:
        """
        Get the argmax of a posterior distribution.

        Parameters
        ----------
        X : NDArray
            Input DataFrame.
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be resampled, and are therefore
            the variables of the optimization

        Returns
        -------
        NDArray
            DataFrame with imputed values.
        """
        X_center = X - self.means
        X_imputed = _conjugate_gradient(self.cov_inv, X_center, mask_na)
        X_imputed = self.means + X_imputed
        return X_imputed

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
        if n_iter < 10:
            return False

        min_diff_means1 = min_diff_Linf(list_covs, n_steps=1)
        min_diff_covs1 = min_diff_Linf(list_means, n_steps=1)
        min_diff_reached = min_diff_means1 < self.tolerance and min_diff_covs1 < self.tolerance

        min_diff_means5 = min_diff_Linf(list_covs, n_steps=5)
        min_diff_covs5 = min_diff_Linf(list_means, n_steps=5)

        min_diff_stable = (
            min_diff_means5 < self.stagnation_threshold
            and min_diff_covs5 < self.stagnation_threshold
        )

        min_diff_loglik5_ord1 = min_diff_Linf(list_logliks, n_steps=5)
        min_diff_loglik5_ord2 = min_diff_Linf(list_logliks, n_steps=5, order=2)
        max_loglik = (min_diff_loglik5_ord1 < self.stagnation_loglik) or (
            min_diff_loglik5_ord2 < self.stagnation_loglik
        )

        return min_diff_reached or min_diff_stable or max_loglik


class VARpEM(EM):
    """
    Imputation of missing values using a vector autoregressive model through EM optimization and
    using a projected Ornstein-Uhlenbeck process. Equations and notations and from the following
    reference, matrices are transposed for consistency:
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis

    X^n+1 = nu + sum_k A_k^T @ X_k^n + G_n @ S

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
    tolerance : float, optional
        Threshold below which a L infinity norm difference indicates the convergence of the
        parameters
    stagnation_threshold : float, optional
        Threshold below which a L infinity norm difference indicates the convergence of the
        parameters
    stagnation_loglik : float, optional
        Threshold below which an absolute difference of the log likelihood indicates the
        convergence of the parameters
    period : int, optional
        Integer used to fold the temporal data periodically
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
    >>> from qolmat.imputations.em_sampler import VARpEM
    >>> imputer = VARpEM(method="sample", random_state=11)
    >>> X = np.array([[1, 1, 1, 1],
    ...               [np.nan, np.nan, 3, 2],
    ...               [1, 2, 2, 1], [2, 2, 2, 2]])
    >>> imputer.fit_transform(X)  # doctest: +SKIP
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
        max_lagp: int = 2,
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
        self.dict_criteria_stop = {"logliks": [], "S": [], "B": []}
        self.p_to_fit = False
        self.p = p  # type: ignore #noqa
        self.max_lagp = max_lagp
        if self.p is None:
            self.p_to_fit = True

    def get_loglikelihood(self, X: NDArray) -> float:
        """
        Value of the log-likelihood up to a constant for the provided X, using the attributes
        `nu`, `B` and `S` for the VAR(p) distribution.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column

        Returns
        -------
        float
            Computed value
        """
        Z, Y = utils.create_lag_matrices(X, self.p)
        U = Y - Z @ self.B
        return -(U @ self.S_inv * U).sum().sum() / 2

    def gradient_X_loglik(self, X: NDArray) -> NDArray:
        """
        Gradient of the log-likelihood for the provided X, using the attributes
        `means` and `cov_inv` for the VAR(p) distribution.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column

        Returns
        -------
        NDArray
            The gradient of the log-likelihood with respect to the input variable `X`.
        """
        n_rows, n_cols = X.shape
        Z, Y = utils.create_lag_matrices(X, p=self.p)
        U = Y - Z @ self.B
        grad_1 = np.zeros(X.shape)
        grad_1[self.p :, :] = -U @ self.S_inv
        grad_2 = np.zeros(X.shape)
        for lag in range(self.p):
            A = self.B[1 + lag * n_cols : 1 + (lag + 1) * n_cols, :]
            grad_2[self.p - lag - 1 : -lag - 1, :] += U @ self.S_inv @ A.T

        return grad_1 + grad_2

    def get_gamma(self) -> NDArray:
        """
        Normalisation matrix used to stabilize the sampling process

        Returns
        -------
        NDArray
            Gamma matrix
        """
        # gamma = np.diagonal(self.S).reshape(1, -1)
        gamma = self.S
        return gamma

    def update_criteria_stop(self, X: NDArray):
        """
        Updates the variables which will be used to compute the stop critera

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column
        """
        self.loglik = self.get_loglikelihood(X)
        self.dict_criteria_stop["S"].append(self.list_S[-1])
        self.dict_criteria_stop["B"].append(self.list_B[-1])
        self.dict_criteria_stop["logliks"].append(self.loglik)

    def reset_learned_parameters(self):
        """
        Resets all lists of estimated parameters before starting a new estimation.
        """
        self.list_ZZ = []
        self.list_ZY = []
        self.list_B = []
        self.list_S = []
        self.list_YY = []

    def update_parameters(self, X: NDArray) -> None:
        """
        Retains statistics relative to the current sample, in prevision of combining them.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column
        """

        Z, Y = utils.create_lag_matrices(X, self.p)
        n_obs = len(Z)
        ZZ = Z.T @ Z / n_obs
        ZZ_inv = np.linalg.pinv(ZZ)
        ZY = Z.T @ Y / n_obs
        B = ZZ_inv @ ZY
        U = Y - Z @ B
        S = U.T @ U / n_obs
        YY = Y.T @ Y / n_obs

        self.list_ZZ.append(ZZ)
        self.list_ZY.append(ZY)
        self.list_B.append(B)
        self.list_S.append(S)
        self.list_YY.append(YY)

    def combine_parameters(self) -> None:
        """
        Combine all statistics computed for each sample in the update step. The estimation of `nu`
         and `B` corresponds to the MLE, whereas `S` is approximated.
        """
        list_ZZ = self.list_ZZ[-self.n_samples :]
        list_ZY = self.list_ZY[-self.n_samples :]
        list_YY = self.list_YY[-self.n_samples :]

        stack_ZZ = np.stack(list_ZZ)
        self.ZZ = np.mean(stack_ZZ, axis=0)
        stack_ZY = np.stack(list_ZY)
        self.ZY = np.mean(stack_ZY, axis=0)
        self.ZZ_inv = np.linalg.pinv(self.ZZ)
        self.B = self.ZZ_inv @ self.ZY
        stack_YY = np.stack(list_YY)
        self.YY = np.mean(stack_YY, axis=0)
        self.S = self.YY - self.ZY.T @ self.B - self.B.T @ self.ZY + self.B.T @ self.ZZ @ self.B
        self.S[self.S < 1e-12] = 0
        self.S_inv = np.linalg.pinv(self.S, rcond=1e-10)

    def _check_convergence(self) -> bool:
        """
        Check if the EM algorithm has converged. Three criteria:
        1) if the differences between the estimates of the parameters (mean and covariance) is
        less than a threshold (min_diff_reached - tolerance).
        OR 2) if the difference of the consecutive differences of the estimates is less than a
        threshold, i.e. stagnates over the last 5 interactions (min_diff_stable -
        stagnation_threshold).
        OR 3) if the likelihood of the data no longer increases,
        i.e. stagnates over the last 5 iterations (max_loglik - stagnation_loglik).

        Returns
        -------
        bool
            True/False if the algorithm has converged
        """

        list_B = self.dict_criteria_stop["B"]
        list_S = self.dict_criteria_stop["S"]
        list_logliks = self.dict_criteria_stop["logliks"]

        n_iter = len(list_B)
        if n_iter < 10:
            return False

        min_diff_B1 = min_diff_Linf(list_B, n_steps=1)
        min_diff_S1 = min_diff_Linf(list_S, n_steps=1)
        min_diff_reached = min_diff_B1 < self.tolerance and min_diff_S1 < self.tolerance

        min_diff_B5 = min_diff_Linf(list_B, n_steps=5)
        min_diff_S5 = min_diff_Linf(list_S, n_steps=5)
        min_diff_stable = (
            min_diff_B5 < self.stagnation_threshold and min_diff_S5 < self.stagnation_threshold
        )

        max_loglik5_ord1 = min_diff_Linf(list_logliks, n_steps=5, order=1)
        max_loglik5_ord2 = min_diff_Linf(list_logliks, n_steps=5, order=2)
        max_loglik = (max_loglik5_ord1 < self.stagnation_loglik) or (
            max_loglik5_ord2 < self.stagnation_loglik
        )

        return min_diff_reached or min_diff_stable or max_loglik
