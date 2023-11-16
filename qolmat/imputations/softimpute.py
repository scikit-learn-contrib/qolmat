from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin

from qolmat.utils import utils
from qolmat.imputations.rpca import rpca_utils


class SoftImpute(BaseEstimator, TransformerMixin):
    """
    This class implements the SoftImpute ALS algorithm presented in
    Hastie, Trevor, et al. "Matrix completion and low-rank SVD
    via fast alternating least squares." The Journal of Machine Learning
    Research 16.1 (2015): 3367-3402.
    min_A,B || Proj(X - AB')||_F^2 + tau * (|| A ||_F^2 + || B ||_F^2)

    Parameters
    ----------
    period : int
        Number of rows of the array if the array is 1D and
        reshaped into a 2D array. Corresponds to the period of the time series,
        if 1D time series is passed.
    rank : int
        Estimated rank of the matrix
    tolerance : float
        Tolerance for the convergence criterion
    tau : float
        regularisation parameter
    max_iterations : int
        Maximum number of iterations
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility
    verbose : bool
        flag for verbosity
    projected : bool
        If true, only imputed values are changed.
        If False, the matrix obtained via the algorithm is returned, by default True

    Examples
    --------
    >>> import numpy as np
    >>> from qolmat.imputations.softimpute import SoftImpute
    >>> X = np.array([[1, 2, np.nan, 4], [1, 5, 3, np.nan], [4, 2, 3, 2], [1, 1, 5, 4]])
    >>> X_imputed = SoftImpute(random_state=11).fit_transform(X)
    >>> print(X_imputed)
    [[1.         2.         3.7242757  4.        ]
     [1.         5.         3.         1.97846028]
     [4.         2.         3.         2.        ]
     [1.         1.         5.         4.        ]]
    """

    def __init__(
        self,
        period: int = 1,
        rank: int = 2,
        tolerance: float = 1e-05,
        tau: float = 0,
        max_iterations: int = 100,
        random_state: Union[None, int, np.random.RandomState] = None,
        verbose: bool = False,
        projected: bool = True,
    ):
        self.period = period
        self.rank = rank
        self.tolerance = tolerance
        self.tau = tau
        self.max_iterations = max_iterations
        self.random_state = sku.check_random_state(random_state)
        self.verbose = verbose
        self.projected = projected
        self.u: NDArray = np.empty(0)
        self.d: NDArray = np.empty(0)
        self.v: NDArray = np.empty(0)

    def fit(self, X: NDArray, y=None) -> SoftImpute:
        """Fit the imputer on X.

        Parameters
        ----------
        X : NDArray
            Input data

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            The fitted `SoftImpute` class instance.
        """
        X_imputed = X.copy()
        X_imputed = utils.prepare_data(X_imputed, self.period)

        if not isinstance(X_imputed, np.ndarray):
            raise AssertionError("Invalid type. X must be a NDArray.")

        n, m = X_imputed.shape
        mask = np.isnan(X_imputed)
        V = np.zeros((m, self.rank))
        U = self.random_state.normal(0.0, 1.0, (n, self.rank))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        Dsq = np.ones((self.rank, 1))
        col_means = np.nanmean(X_imputed, axis=0)
        np.copyto(X_imputed, col_means, where=np.isnan(X_imputed))
        if self.rank is None:
            self.rank = rpca_utils.approx_rank(X_imputed)
        for iter_ in range(self.max_iterations):
            U_old = U
            V_old = V
            Dsq_old = Dsq

            B = U.T @ X_imputed
            if self.tau > 0:
                tmp = Dsq / (Dsq + self.tau)
                B = B * tmp
            Bsvd = np.linalg.svd(B.T, full_matrices=False)
            V = Bsvd[0]
            Dsq = Bsvd[1][:, np.newaxis]
            U = U @ Bsvd[2]
            tmp = Dsq * V.T
            X_hat = U @ tmp
            X_imputed[mask] = X_hat[mask]

            A = (X_imputed @ V).T
            if self.tau > 0:
                tmp = Dsq / (Dsq + self.tau)
                A = A * tmp
            Asvd = np.linalg.svd(A.T, full_matrices=False)
            U = Asvd[0]
            Dsq = Asvd[1][:, np.newaxis]
            V = V @ Asvd[2]
            tmp = Dsq * V.T
            X_hat = U @ tmp
            X_imputed[mask] = X_hat[mask]

            ratio = self._check_convergence(U_old, Dsq_old, V_old, U, Dsq, V)
            if self.verbose:
                print(f"iter {iter_}: ratio = {round(ratio, 4)}")
            if ratio < self.tolerance:
                break

        self.u = U[:, : self.rank]
        self.d = Dsq[: self.rank]
        self.v = V[:, : self.rank]

        return self

    def _check_convergence(
        self,
        U_old: NDArray,
        Ds_qold: NDArray,
        V_old: NDArray,
        U: NDArray,
        Dsq: NDArray,
        V: NDArray,
    ) -> float:
        """Given a pair of iterates (U_old, Ds_qold, V_old) and (U, Dsq, V),
        it computes the relative change in Frobenius norm given by
        || U_old @  Dsq_old @ V_old.T - U @  Dsq @ V.T ||_F^2
        / || U_old @  Ds_qold @ V_old.T ||_F^2

        Parameters
        ----------
        U_old : NDArray
            previous matrix U
        Ds_qold : NDArray
            previous matrix Dsq
        V_old : NDArray
            previous matrix V
        U : NDArray
            current matrix U
        Dsq : NDArray
            current matrix Dsq
        V : NDArray
            current matrix V

        Returns
        -------
        float
            relative change
        """
        if any(arg is None for arg in (U_old, Ds_qold, V_old, U, Dsq, V)):
            raise ValueError("One or more arguments are None.")

        denom = (Ds_qold**2).sum()
        utu = Dsq * (U.T @ U_old)
        vtv = Ds_qold * (V_old.T @ V)
        uvprod = (utu @ vtv).diagonal().sum()
        num = denom + (Ds_qold**2).sum() - 2 * uvprod
        return num / max(denom, 1e-9)

    def transform(self, X: NDArray) -> NDArray:
        """Impute all missing values in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X : NDArray
            The imputed dataset.
        """
        X_transformed = self.u @ np.diag(self.d.T[0]) @ (self.v).T
        if self.projected:
            X_ = utils.prepare_data(X, self.period)
            mask = np.isnan(X_)
            X_transformed[~mask] = X_[~mask]

        X_transformed = utils.get_shape_original(X_transformed, X.shape)

        if np.all(np.isnan(X_transformed)):
            raise AssertionError("Result contains NaN. This is a bug.")

        return X_transformed
