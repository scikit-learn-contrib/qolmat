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
    min_L,Q || Proj(D - LQ')||_F^2 + tau * (|| L ||_F^2 + || Q ||_F^2)

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
    >>> D = np.array([[1, 2, np.nan, 4], [1, 5, 3, np.nan], [4, 2, 3, 2], [1, 1, 5, 4]])
    >>> D_imputed = SoftImpute(random_state=11).fit_transform(D)
    >>> print(D_imputed)
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

    def fit(self, D: NDArray, y=None) -> SoftImpute:
        """Fit the imputer on D.

        Parameters
        ----------
        D : NDArray
            Input data

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            The fitted `SoftImpute` class instance.
        """
        D_imputed = D.copy()
        D_imputed = utils.prepare_data(D_imputed, self.period)

        if not isinstance(D_imputed, np.ndarray):
            raise AssertionError("Invalid type. D must be a NDArray.")

        n, m = D_imputed.shape
        mask = np.isnan(D_imputed)
        V = np.zeros((m, self.rank))
        U = self.random_state.normal(0.0, 1.0, (n, self.rank))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        Dsq = np.ones((self.rank, 1))
        col_means = np.nanmean(D_imputed, axis=0)
        np.copyto(D_imputed, col_means, where=np.isnan(D_imputed))
        if self.rank is None:
            self.rank = rpca_utils.approx_rank(D_imputed)
        for iter_ in range(self.max_iterations):
            U_old = U
            V_old = V
            Dsq_old = Dsq

            Q = U.T @ D_imputed
            if self.tau > 0:
                tmp = Dsq / (Dsq + self.tau)
                Q = Q * tmp
            Bsvd = np.linalg.svd(Q.T, full_matrices=False)
            V = Bsvd[0]
            Dsq = Bsvd[1][:, np.newaxis]
            U = U @ Bsvd[2]
            tmp = Dsq * V.T
            D_hat = U @ tmp
            D_imputed[mask] = D_hat[mask]

            L = (D_imputed @ V).T
            if self.tau > 0:
                tmp = Dsq / (Dsq + self.tau)
                L = L * tmp
            Lsvd = np.linalg.svd(L.T, full_matrices=False)
            U = Lsvd[0]
            Dsq = Lsvd[1][:, np.newaxis]
            V = V @ Lsvd[2]
            tmp = Dsq * V.T
            D_hat = U @ tmp
            D_imputed[mask] = D_hat[mask]

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

    def transform(self, D: NDArray) -> NDArray:
        """Impute all missing values in D.

        Parameters
        ----------
        D : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        D : NDArray
            The imputed dataset.
        """
        D_transformed = self.u @ np.diag(self.d.T[0]) @ (self.v).T
        if self.projected:
            D_ = utils.prepare_data(D, self.period)
            mask = np.isnan(D_)
            D_transformed[~mask] = D_[~mask]

        D_transformed = utils.get_shape_original(D_transformed, D.shape)

        if np.all(np.isnan(D_transformed)):
            raise AssertionError("Result contains NaN. This is a bug.")

        return D_transformed
