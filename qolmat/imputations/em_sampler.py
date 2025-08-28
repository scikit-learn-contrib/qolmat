"""Script for EM imputation."""

import logging
import warnings
from abc import abstractmethod
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as spl
from scipy import optimize as spo
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin

from qolmat.utils import utils

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _conjugate_gradient(A: NDArray, X: NDArray, mask: NDArray) -> NDArray:
    """Compute conjugate gradient.

    Minimize Tr(X.T AX) wrt X where X is constrained to the initial value
    outside the given mask To this aim, we compute in parallel a gradient
    algorithm for each row.

    Parameters
    ----------
    A : NDArray
        Symmetrical matrix defining the quadratic minimization problem
    X : NDArray
        Array containing the values to optimize
    mask : NDArray
        Boolean array indicating if a value of X is a variable of
        the optimization

    Returns
    -------
    NDArray
        Minimized array.

    """
    rows_imputed = mask.any(axis=1)
    X_temp = X[rows_imputed, :].copy()
    mask = mask[rows_imputed, :].copy()
    n_iter = mask.sum(axis=1).max()
    n_rows, n_cols = X_temp.shape
    X_temp[mask] = 0
    b = -X_temp @ A
    b[~mask] = 0
    xn, pn, rn = np.zeros((n_rows, n_cols)), b, b  # Initialisation
    alphan = np.zeros(n_rows)
    betan = np.zeros(n_rows)
    for n in range(n_iter + 2):
        # if np.max(np.sum(rn**2)) < tolerance :
        #     X_temp[mask_isna] = xn[mask_isna]
        #     return X_temp.transpose()
        Apn = pn @ A
        Apn[~mask] = 0
        numerator = np.sum(rn**2, axis=1)
        denominator = np.sum(pn * Apn, axis=1)
        not_converged = denominator != 0
        # we stop updating if convergence is reached for this row
        alphan[not_converged] = (
            numerator[not_converged] / denominator[not_converged]
        )

        xn, rnp1 = xn + pn * alphan[:, None], rn - Apn * alphan[:, None]
        numerator = np.sum(rnp1**2, axis=1)
        denominator = np.sum(rn**2, axis=1)
        not_converged = denominator != 0
        # we stop updating if convergence is reached for this row
        betan[not_converged] = (
            numerator[not_converged] / denominator[not_converged]
        )

        pn, rn = rnp1 + pn * betan[:, None], rnp1

    X_temp[mask] = xn[mask]
    X_final = X.copy()
    X_final[rows_imputed, :] = X_temp

    return X_final


def max_diff_Linf(
    list_params: List[NDArray], n_steps: int, order: int = 1
) -> float:
    """Compute the maximal L infinity norm.

    Computed between the `n_steps` last elements spaced by order.
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
    """Abstract class for EM imputatoin.

    It uses imputation through EM optimization and
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
        Number of data samples used to estimate the parameters of the
        distribution. Default, 10
    ampli : float, optional
        Whether to sample the posterior (1)
        or to maximise likelihood (0), by default 1.
    random_state : int, optional
        The seed of the pseudo random number generator to use,
        for reproductibility.
    dt : float, optional
        Process integration time step, a large value increases the sample bias
        and can make the algorithm unstable, but compensates for a
        smaller n_iter_ou. By default, 2e-2.
    tolerance : float, optional
        Threshold below which a L infinity norm difference indicates the
        convergence of the parameters
    stagnation_threshold : float, optional
        Threshold below which a stagnation of the L infinity norm difference
        indicates the convergence of the parameters
    stagnation_loglik : float, optional
        Threshold below which an absolute difference of the log likelihood
        indicates the convergence of the parameters
    min_std: float, optional
        Threshold below which the initial data matrix is considered
        ill-conditioned
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
        min_std: float = 1e-6,
        period: int = 1,
        verbose: bool = False,
    ):
        if method not in ["mle", "sample"]:
            raise ValueError(
                "`method` must be 'mle' or 'sample', "
                f"provided value is '{method}'."
            )

        self.method = method
        self.max_iter_em = max_iter_em
        self.n_iter_ou = n_iter_ou
        self.ampli = ampli
        self.rng = sku.check_random_state(random_state)
        self.dt = dt
        self.tolerance = tolerance
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_loglik = stagnation_loglik

        self.min_std = min_std

        self.dict_criteria_stop: Dict[str, List] = {}
        self.period = period
        self.verbose = verbose
        self.n_samples = n_samples
        self.hash_fit = 0
        self.shape = (0, 0)

    def _check_convergence(self) -> bool:
        return False

    @abstractmethod
    def reset_learned_parameters(self):
        """Reset learned parameters."""
        pass

    @abstractmethod
    def update_parameters(self, X: NDArray):
        """Update parameters."""
        pass

    @abstractmethod
    def combine_parameters(self):
        """Combine parameters."""
        pass

    def fit_parameters(self, X: NDArray):
        """Fir parameters.

        Parameters
        ----------
        __________
        X: NDArray
            Array to compute the parameters.

        """
        self.reset_learned_parameters()
        self.update_parameters(X)
        self.combine_parameters()

    def fit_parameters_with_missingness(self, X: NDArray):
        """Fit the first estimation of the model parameters.

        It is based on data with missing values.

        Parameters
        ----------
        X : NDArray
            Data matrix with missingness

        """
        X_imp = self.init_imputation(X)
        self.fit_parameters(X_imp)

    def update_criteria_stop(self, X: NDArray):
        """Update the stopping criteria based on X.

        Parameters
        ----------
        X : NDArray
            array used to compute log likelihood.

        """
        self.loglik = self.get_loglikelihood(X)

    @abstractmethod
    def get_loglikelihood(self, X: NDArray) -> float:
        """Compute the loglikelihood of an array.

        Parameters
        ----------
        X : NDArray
            Input array.

        Returns
        -------
        float
            log-likelihood.

        """
        return 0

    @abstractmethod
    def gradient_X_loglik(
        self,
        X: NDArray,
    ) -> NDArray:
        """Compute the gradient X loglik.

        Parameters
        ----------
        X : NDArray
            input array

        Returns
        -------
        NDArray
            gradient

        """
        return np.empty  # type: ignore #noqa

    def get_gamma(self, n_cols: int) -> NDArray:
        """Get gamma.

        Normalization matrix in the sampling process.

        Parameters
        ----------
        n_cols : int
            Number of variables in the data matrix

        Returns
        -------
        NDArray
            Gamma matrix

        """
        # return np.ones((1, n_cols))
        return np.eye(n_cols)

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray) -> NDArray:
        """Get the argmax of a posterior distribution using the BFGS algorithm.

        Parameters
        ----------
        X : NDArray
            Input numpy array without missingness
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be
            resampled, and are therefore the variables of the optimization

        Returns
        -------
        NDArray
            DataFrame with imputed values.

        """

        def fun_obj(x):
            x_mat = X.copy()
            x_mat[mask_na] = x
            return -self.get_loglikelihood(x_mat)

        def fun_jac(x):
            x_mat = X.copy()
            x_mat[mask_na] = x
            grad_x = -self.gradient_X_loglik(x_mat)
            grad_x = grad_x[mask_na]
            return grad_x

        # the method BFGS is much slower, probabily not adapted
        # to the high-dimension setting
        res = spo.minimize(fun_obj, X[mask_na], jac=fun_jac, method="CG")
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
        """Sample the Gaussian distribution.

        Under the constraint that not na values must remain
        unchanged, using a projected Ornstein-Uhlenbeck process.
        The sampled distribution tends to the target distribution
        in the limit dt -> 0 and n_iter_ou x dt -> infty.

        Parameters
        ----------
        X : NDArray
            Inital dataframe to be imputed, which should have been already
            imputed using a simple method. This first imputation will be used
            as an initial guess.
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be
            resampled.
        estimate_params : bool
            Indicates if the parameters of the distribution should be estimated
            while the data are sampled.

        Returns
        -------
        NDArray
            Sampled data matrix

        """
        X_copy = X.copy()
        n_rows, n_cols = X_copy.shape
        if estimate_params:
            self.reset_learned_parameters()
        X_init = X.copy()
        gamma = self.get_gamma(n_cols)
        sqrt_gamma = np.real(spl.sqrtm(gamma))

        for i in range(self.n_iter_ou):
            noise = self.ampli * self.rng.normal(0, 1, size=(n_rows, n_cols))
            grad_X = -self.gradient_X_loglik(X_copy)
            X_copy += (
                -self.dt * grad_X @ gamma
                + np.sqrt(2 * self.dt) * noise @ sqrt_gamma
            )
            X_copy[~mask_na] = X_init[~mask_na]
            if estimate_params:
                self.update_parameters(X_copy)

        return X_copy

    def fit_X(self, X: NDArray) -> None:
        """Ft X array.

        Parameters
        ----------
        X : NDArray
            Input array.

        """
        mask_na = np.isnan(X)

        # first imputation
        X_imp = self.init_imputation(X)
        self._check_conditionning(X_imp)

        self.fit_parameters_with_missingness(X)

        if not np.any(mask_na):
            self.X = X
            return

        X = self._maximize_likelihood(X_imp, mask_na)

        for iter_em in range(self.max_iter_em):
            X = self._sample_ou(X, mask_na)

            self.combine_parameters()

            # Stop criteria
            self.update_criteria_stop(X)
            if self._check_convergence():
                if self.verbose:
                    logging.info(f"EM converged after {iter_em} iterations.")
                break

        self.dict_criteria_stop = {key: [] for key in self.dict_criteria_stop}
        self.X = X

    def fit(self, X: NDArray) -> "EM":
        """Fit the statistical distribution with the input X array.

        Parameters
        ----------
        X : NDArray
            Numpy array to be imputed

        """
        X = X.copy()
        # utils.check_dtypes(X)
        # sku.check_array(X, ensure_all_finite="allow-nan", dtype="float")
        sku.validation.validate_data(
            self, X, ensure_all_finite="allow-nan", dtype="float"
        )
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
        """Transform the input X array by imputing the missing values.

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
        X = X.copy()
        # sku.check_array(X, ensure_all_finite="allow-nan", dtype="float")
        sku.validation.validate_data(
            self, X, ensure_all_finite="allow-nan", dtype="float", reset=False
        )

        # shape_original = X.shape
        if hash(X.tobytes()) == self.hash_fit:
            X = self.X
            warm_start = True
        else:
            X = utils.prepare_data(X, self.period)
            X = self.init_imputation(X)
            warm_start = False

        X, mask_na = self.pretreatment(X, mask_na)

        if (self.method == "mle") or not warm_start:
            X = self._maximize_likelihood(X, mask_na)
        if self.method == "sample":
            X = self._sample_ou(X, mask_na, estimate_params=False)

        if np.all(np.isnan(X)):
            raise AssertionError("Result contains NaN. This is a bug.")

        return X

    def pretreatment(self, X, mask_na) -> Tuple[NDArray, NDArray]:
        """Pretreat the data before imputation by EM, making it more robust.

        Parameters
        ----------
        X : NDArray
            Data matrix without nans
        mask_na : NDArray
            Boolean matrix indicating which entries are to be imputed

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing:
            - X the pretreatd data matrix
            - mask_na the updated mask

        """
        return X, mask_na

    def _check_conditionning(self, X: NDArray):
        """Check that the data matrix X is not ill-conditioned.

        Running the EM algorithm on data with colinear columns leads to
        numerical instability and unconsistent results.

        Parameters
        ----------
        X : NDArray
            Data matrix

        Raises
        ------
        IllConditioned
            Data matrix is ill-conditioned due to colinear columns.

        """
        n_samples, n_cols = X.shape
        # if n_rows == 1 the function np.cov returns a float
        if n_samples == 1:
            raise ValueError("EM cannot be fitted when n_samples = 1!")

        cov = np.cov(X, bias=True, rowvar=False).reshape(n_cols, -1)
        _, sv, _ = spl.svd(cov)
        min_sv = min(np.sqrt(sv))
        if min_sv < self.min_std:
            warnings.warn(
                "The covariance matrix is ill-conditioned, "
                "indicating high-colinearity: the "
                "smallest singular value of the data matrix is smaller "
                "than the threshold "
                f"min_std ({min_sv} < {self.min_std}). "
                "Consider removing columns of decreasing the threshold."
            )


class MultiNormalEM(EM):
    """Multinormal EM imputer.

    Imputation of missing values using a multivariate Gaussian model through
    EM optimization and using a projected Ornstein-Uhlenbeck process.

    Parameters
    ----------
    method : Literal["mle", "sample"]
        Method for imputation, choose among "sample" or "mle".
    max_iter_em : int, optional
        Maximum number of steps in the EM algorithm
    n_iter_ou : int, optional
        Number of iterations for the Gibbs sampling method (+ noise addition),
        necessary for convergence, by default 50.
    n_samples : int, optional
        Number of data samples used to estimate the parameters of the
        distribution. Default, 10
    ampli : float, optional
        Whether to sample the posterior (1)
        or to maximise likelihood (0), by default 1.
    random_state : int, optional
        The seed of the pseudo random number generator to use,
        for reproductibility.
    dt : float
        Process integration time step, a large value increases the sample bias
        and can make the algorithm unstable, but compensates for a
        smaller n_iter_ou. By default, 2e-2.
    tolerance : float, optional
        Threshold below which a L infinity norm difference indicates the
        convergence of the parameters
    stagnation_threshold : float, optional
        Threshold below which a L infinity norm difference indicates the
        convergence of the parameters
    stagnation_loglik : float, optional
        Threshold below which an absolute difference of the log likelihood
        indicates the convergence of the parameters
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
        n_samples: int = 10,
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
            n_samples=n_samples,
            ampli=ampli,
            random_state=random_state,
            dt=dt,
            tolerance=tolerance,
            stagnation_threshold=stagnation_threshold,
            stagnation_loglik=stagnation_loglik,
            period=period,
            verbose=verbose,
        )
        self.cov = np.array([[]])
        self.dict_criteria_stop = {"logliks": [], "means": [], "covs": []}

    def get_loglikelihood(self, X: NDArray) -> float:
        """Get the log-likelihood.

        Value of the log-likelihood up to a constant for the provided X,
        using the attributes `means` and `cov_inv` for the multivariate
        normal distribution.

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
        """Compute the gradient of the log-likelihood for the provided X.

        It uses  the attributes
        `means` and `cov_inv` for the multivariate normal distribution.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column

        Returns
        -------
        NDArray
            The gradient of the log-likelihood with respect to the input
            variable `X`.

        """
        grad_X = -(X - self.means) @ self.cov_inv
        return grad_X

    def get_gamma(self, n_cols: int) -> NDArray:
        """Get gamma.

        If the covariance matrix is not full-rank, defines the
        projection matrix keeping the sampling process in the relevant
        subspace.

        Parameters
        ----------
        n_cols : int
            Number of variables in the data matrix

        Returns
        -------
        NDArray
            Gamma matrix

        """
        U, diag, Vt = spl.svd(self.cov)
        diag_trunc = np.where(diag < self.min_std**2, 0, diag)
        diag_trunc = np.where(diag_trunc == 0, 0, np.min(diag_trunc))

        gamma = (U * diag_trunc) @ Vt
        # gamma = np.eye(len(self.cov))

        return gamma

    def update_criteria_stop(self, X: NDArray):
        """Update the variables to compute the stopping critera.

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
        """Reset lists of parameters before starting a new estimation."""
        self.list_means = []
        self.list_cov = []

    def update_parameters(self, X):
        """Retain statistics relative to the current sample.

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
        """Combine all statistics computed for each sample in the update step.

        If uses the MANOVA formula.
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

    def fit_parameters_with_missingness(self, X: NDArray):
        """Fit the first estimation of the model parameters.

        It is based on data with missing values.

        Parameters
        ----------
        X : NDArray
            Data matrix with missingness

        """
        self.means, self.cov = utils.nan_mean_cov(X)
        self.cov_inv = np.linalg.pinv(self.cov)

    def set_parameters(self, means: NDArray, cov: NDArray):
        """Set the model parameters from a user value.

        Parameters
        ----------
        means : NDArray
            Specified value for the mean vector
        cov : NDArray
            Specified value for the covariance matrix

        """
        self.means = means
        self.cov = cov
        self.cov_inv = np.linalg.pinv(self.cov)

    def _maximize_likelihood(self, X: NDArray, mask_na: NDArray) -> NDArray:
        """Get the argmax of a posterior distribution.

        Parameters
        ----------
        X : NDArray
            Input DataFrame without missingness
        mask_na : NDArray
            Boolean dataframe indicating which coefficients should be
            resampled, and are therefore the variables of the optimization

        Returns
        -------
        NDArray
            DataFrame with imputed values.

        """
        X_center = X - self.means
        X_imputed = _conjugate_gradient(self.cov_inv, X_center, mask_na)
        X_imputed = self.means + X_imputed
        return X_imputed

    def init_imputation(self, X: NDArray) -> NDArray:
        """First simple imputation before iterating.

        Parameters
        ----------
        X : NDArray
            Data matrix, with missing values

        Returns
        -------
        NDArray
            Imputed matrix

        """
        return utils.impute_nans(X, method="median")

    def _check_convergence(self) -> bool:
        """Check if the EM algorithm has converged.

        Three criteria:
        1) if the differences between the estimates of the parameters
        (mean and covariance) is less than a threshold
        (min_diff_reached - tolerance).
        2) if the difference of the consecutive differences of the estimates
        is less than a threshold, i.e. stagnates over the last 5 interactions
        (min_diff_stable - stagnation_threshold).
        3) if the likelihood of the data no longer increases,
        i.e. stagnates over the last 5 iterations
        (max_loglik - stagnation_loglik).

        Returns
        -------
        bool
            True/False if the algorithm has converged

        """
        list_means = self.dict_criteria_stop["means"]
        list_covs = self.dict_criteria_stop["covs"]
        list_logliks = self.dict_criteria_stop["logliks"]

        n_iter = len(list_means)
        if n_iter < 3:
            return False

        min_diff_means1 = max_diff_Linf(list_means, n_steps=1)
        min_diff_covs1 = max_diff_Linf(list_covs, n_steps=1)
        min_diff_reached = (
            min_diff_means1 < self.tolerance
            and min_diff_covs1 < self.tolerance
        )

        if min_diff_reached:
            return True

        if n_iter < 7:
            return False

        min_diff_means5 = max_diff_Linf(list_means, n_steps=5)
        min_diff_covs5 = max_diff_Linf(list_covs, n_steps=5)

        min_diff_stable = (
            min_diff_means5 < self.stagnation_threshold
            and min_diff_covs5 < self.stagnation_threshold
        )

        min_diff_loglik5_ord1 = max_diff_Linf(list_logliks, n_steps=5)
        min_diff_loglik5_ord2 = max_diff_Linf(list_logliks, n_steps=5, order=2)
        max_loglik = (min_diff_loglik5_ord1 < self.stagnation_loglik) or (
            min_diff_loglik5_ord2 < self.stagnation_loglik
        )
        return min_diff_stable or max_loglik


class VARpEM(EM):
    """VAR(p) EM imputer.

    Imputation of missing values using a vector autoregressive model through
    EM optimization and using a projected Ornstein-Uhlenbeck process.
    Equations and notations and from the following
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
        The seed of the pseudo random number generator to use,
        for reproductibility.
    dt : float
        Process integration time step, a large value increases the sample bias
        and can make the algorithm unstable, but compensates for
        a smaller n_iter_ou. By default, 2e-2.
    tolerance : float, optional
        Threshold below which a L infinity norm difference indicates
        the convergence of the parameters
    stagnation_threshold : float, optional
        Threshold below which a L infinity norm difference indicates the
        convergence of the parameters
    stagnation_loglik : float, optional
        Threshold below which an absolute difference of the log likelihood
        indicates the convergence of the parameters
    period : int, optional
        Integer used to fold the temporal data periodically
    verbose: bool
        default `False`

    Attributes
    ----------
    X_intermediate : list
        List of pd.DataFrame giving the results of the EM process as function
        of the iteration number.

    Examples
    --------
    >>> import numpy as np
    >>> from qolmat.imputations.em_sampler import VARpEM
    >>> imputer = VARpEM(method="sample", random_state=11)
    >>> X = np.array(
    ...     [[1, 1, 1, 1], [np.nan, np.nan, 3, 2], [1, 2, 2, 1], [2, 2, 2, 2]]
    ... )
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
        """Get the log-likelihood.

        Value of the log-likelihood up to a constant for the provided X,
        using the attributes `nu`, `B` and `S` for the VAR(p) distribution.

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
        """Compute the  gradient of the log-likelihood for the provided X.

        It uses the attributes `means` and `cov_inv`
        for the VAR(p) distribution.

        Parameters
        ----------
        X : NDArray
            Input matrix with variables in column

        Returns
        -------
        NDArray
            The gradient of the log-likelihood with respect
            to the input variable `X`.

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

    def get_gamma(self, n_cols: int) -> NDArray:
        """Compue gamma.

        If the noise matrix is not full-rank, defines the projection matrix
        keeping the sampling process in the relevant subspace.
        Rescales the process to avoid instabilities.

        Parameters
        ----------
        n_cols : int
            Number of variables in the data matrix

        Returns
        -------
        NDArray
            Gamma matrix

        """
        U, diag, Vt = spl.svd(self.S)
        diag_trunc = np.where(diag < self.min_std**2, 0, diag)
        diag_trunc = np.where(diag_trunc == 0, 0, np.min(diag_trunc))

        gamma = (U * diag_trunc) @ Vt
        # gamma = np.eye(len(self.cov))

        return gamma

    def update_criteria_stop(self, X: NDArray):
        """Update the variable to compute the stopping critera.

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
        """Reset lists of parameters before starting a new estimation."""
        self.list_ZZ = []
        self.list_ZY = []
        self.list_B = []
        self.list_S = []
        self.list_YY = []

    def update_parameters(self, X: NDArray) -> None:
        """Retain statistics relative to the current sample.

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
        """Combine statistics computed for each sample in the update step.

        The estimation of `nu` and `B` corresponds to the MLE,
        whereas `S` is approximated.
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
        self.S = (
            self.YY
            - self.ZY.T @ self.B
            - self.B.T @ self.ZY
            + self.B.T @ self.ZZ @ self.B
        )
        self.S[np.abs(self.S) < 1e-12] = 0
        self.S_inv = np.linalg.pinv(self.S, rcond=1e-10)

    def set_parameters(self, B: NDArray, S: NDArray):
        """Set the model parameters from a user value.

        Parameters
        ----------
        B : NDArray
            Specified value for the autoregression matrix
        S : NDArray
            Specified value for the noise covariance matrix

        """
        self.B = B
        self.S = S
        self.S_inv = np.linalg.pinv(self.S)

    def init_imputation(self, X: NDArray) -> NDArray:
        """First simple imputation before iterating.

        Parameters
        ----------
        X : NDArray
            Data matrix, with missing values

        Returns
        -------
        NDArray
            Imputed matrix

        """
        return utils.linear_interpolation(X)

    def pretreatment(self, X, mask_na) -> Tuple[NDArray, NDArray]:
        """Pretreat the data before imputation by EM, making it more robust.

        In the case of the
        VAR(p) model we freeze the naive imputation on the first observations
        if all variables are missing to avoid explosive imputations.

        Parameters
        ----------
        X : NDArray
            Data matrix without nans
        mask_na : NDArray
            Boolean matrix indicating which entries are to be imputed

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing:
            - X the pretreatd data matrix
            - mask_na the updated mask

        """
        if self.p == 0:
            return X, mask_na
        mask_na = mask_na.copy()
        n_holes_left = np.sum(~np.cumsum(~mask_na, axis=0).any(axis=1))
        mask_na[:n_holes_left] = False
        return X, mask_na

    def _check_convergence(self) -> bool:
        """Check if the EM algorithm has converged.

        Three criteria:
        1) if the differences between the estimates of the parameters
        (mean and covariance) is less than a threshold
        (min_diff_reached - tolerance).
        OR 2) if the difference of the consecutive differences of the
        estimates is less than a threshold, i.e. stagnates over the
        last 5 interactions (min_diff_stable - stagnation_threshold).
        OR 3) if the likelihood of the data no longer increases,
        i.e. stagnates over the last 5 iterations
        (max_loglik - stagnation_loglik).

        Returns
        -------
        bool
            True/False if the algorithm has converged

        """
        list_B = self.dict_criteria_stop["B"]
        list_S = self.dict_criteria_stop["S"]
        list_logliks = self.dict_criteria_stop["logliks"]

        n_iter = len(list_B)
        if n_iter < 3:
            return False

        min_diff_B1 = max_diff_Linf(list_B, n_steps=1)
        min_diff_S1 = max_diff_Linf(list_S, n_steps=1)
        min_diff_reached = (
            min_diff_B1 < self.tolerance and min_diff_S1 < self.tolerance
        )

        if min_diff_reached:
            return True

        if n_iter < 7:
            return False

        min_diff_B5 = max_diff_Linf(list_B, n_steps=5)
        min_diff_S5 = max_diff_Linf(list_S, n_steps=5)
        min_diff_stable = (
            min_diff_B5 < self.stagnation_threshold
            and min_diff_S5 < self.stagnation_threshold
        )

        max_loglik5_ord1 = max_diff_Linf(list_logliks, n_steps=5, order=1)
        max_loglik5_ord2 = max_diff_Linf(list_logliks, n_steps=5, order=2)
        max_loglik = (max_loglik5_ord1 < self.stagnation_loglik) or (
            max_loglik5_ord2 < self.stagnation_loglik
        )
        return min_diff_stable or max_loglik
