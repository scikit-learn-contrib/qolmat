from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn import utils as sku
from sklearn.base import BaseEstimator, TransformerMixin

from qolmat.utils import utils
from qolmat.imputations.rpca import rpca_utils


class SoftImpute(BaseEstimator, TransformerMixin):
    """
    This class implements the Rank Restricted Soft SVD algorithm presented in
    Hastie, Trevor, et al. "Matrix completion and low-rank SVD
    via fast alternating least squares." The Journal of Machine Learning
    Research 16.1 (2015): 3367-3402, Algorithm 2.1
    Given X the input matrix, we solve for the following problem:
    min_A, B || Proj(X - AB')||_F^2 + tau * (|| A ||_F^2 + || B ||_F^2)

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

    Examples
    --------
    >>> import numpy as np
    >>> from qolmat.imputations.softimpute import SoftImpute
    >>> D = np.array([[1, 2, np.nan, 4], [1, 5, 3, np.nan], [4, 2, 3, 2], [1, 1, 5, 4]])
    >>> D = SoftImpute(random_state=11).fit_transform(D)
    >>> print(D)
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
    ):
        self.period = period
        self.rank = rank
        self.tolerance = tolerance
        self.tau = tau
        self.max_iterations = max_iterations
        self.random_state = sku.check_random_state(random_state)
        self.verbose = verbose

    # def decompose(self, X: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
    #     """
    #     Compute the Soft Impute decomposition

    #     Parameters
    #     ----------
    #     D : NDArray
    #         Matrix of the observations
    #     Omega: NDArray
    #         Matrix of missingness, with boolean data

    #     Returns
    #     -------
    #     M: NDArray
    #         Low-rank signal
    #     A: NDArray
    #         Anomalies
    #     """
    #     print()
    #     print()
    #     print(X.shape)
    #     print()
    #     X = utils.linear_interpolation(X)

    #     n, m = X.shape
    #     V = np.zeros((m, self.rank))
    #     U = self.random_state.normal(0.0, 1.0, (n, self.rank))
    #     U, _, _ = np.linalg.svd(U, full_matrices=False)
    #     D2 = np.ones((self.rank, 1))
    #     col_means = np.nanmean(X, axis=0)
    #     np.copyto(X, col_means, where=~Omega)
    #     if self.rank is None:
    #         self.rank = rpca_utils.approx_rank(X)
    #     for iter_ in range(self.max_iterations):
    #         U_old = U
    #         V_old = V
    #         D2_old = D2

    #         BDt = U.T @ X
    #         if self.tau > 0:
    #             BDt *= D2 / (D2**2 + self.tau)
    #         Vtilde, D2tilde, Rt = np.linalg.svd(BDt.T, full_matrices=False)
    #         V = Vtilde
    #         D2 = D2tilde.reshape(-1, 1)
    #         U = U @ Rt
    #         X_hat = U @ (D2 * V.T)
    #         X[~Omega] = X_hat[~Omega]

    #         A = (X @ V).T
    #         if self.tau > 0:
    #             A *= D2 / (D2 + self.tau)
    #         Lsvd = np.linalg.svd(A.T, full_matrices=False)
    #         U = Lsvd[0]
    #         D2 = Lsvd[1][:, np.newaxis]
    #         V = V @ Lsvd[2]
    #         X_hat = U @ (D2 * V.T)
    #         X[~Omega] = X_hat[~Omega]

    #         ratio = self._check_convergence(U_old, D2_old, V_old, U, D2, V)
    #         if self.verbose:
    #             print(f"iter {iter_}: ratio = {round(ratio, 4)}")
    #         if ratio < self.tolerance:
    #             break

    #     u = U[:, : self.rank]
    #     d = D2[: self.rank]
    #     v = V[:, : self.rank]

    #     M = u @ np.diag(d.T[0]) @ (v).T
    #     A = X - M

    #     return M, A

    def decompose(self, X: NDArray, Omega: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Compute the Soft Impute decomposition

        Parameters
        ----------
        D : NDArray
            Matrix of the observations
        Omega: NDArray
            Matrix of missingness, with boolean data

        Returns
        -------
        M: NDArray
            Low-rank signal
        A: NDArray
            Anomalies
        """
        assert self.tau > 0
        if self.rank is None:
            self.rank = rpca_utils.approx_rank(X)
        # X = utils.linear_interpolation(X)

        # Step 1 : Initializing
        n, m = X.shape
        V = np.zeros((m, self.rank))
        U = self.random_state.normal(0.0, 1.0, (n, self.rank))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        D = np.ones((1, self.rank))
        # col_means = np.nanmean(X, axis=0)
        # np.copyto(X, col_means, where=~Omega)

        A = U * D
        B = V * D
        for iter_ in range(self.max_iterations):
            U_old = U
            V_old = V
            D_old = D

            # Step 2 : Upate on B
            D2_invreg = (D**2 + self.tau) ** (-1)
            Btilde = ((U * D).T @ np.where(Omega, X - A @ B.T, 0) + (B * D**2).T).T
            Btilde = Btilde * D2_invreg

            Utilde, D2tilde, _ = np.linalg.svd(Btilde * D, full_matrices=False)
            V = Utilde
            D = np.sqrt(D2tilde).reshape(1, -1)
            B = V * D

            # Step 3 : Upate on A
            D2_invreg = (D**2 + self.tau) ** (-1)
            Atilde = ((V * D).T @ np.where(Omega, X.T - B @ A.T, 0) + (A * D**2).T).T
            Atilde = Atilde * D2_invreg

            Utilde, D2tilde, _ = np.linalg.svd(Atilde * D, full_matrices=False)
            U = Utilde
            D = np.sqrt(D2tilde).reshape(1, -1)
            A = U * D

            # Step 4 : Stopping upon convergence
            ratio = self._check_convergence(U_old, D_old, V_old, U, D, V)
            if self.verbose:
                print(f"Iteration {iter_}: ratio = {round(ratio, 4)}")
            if ratio < self.tolerance:
                print(f"Convergence reached at iteration {iter_} with ratio = {round(ratio, 4)}")
                break

        Xstar = np.where(Omega, X - A @ B.T, 0) + A @ B.T
        M = Xstar @ V
        U, D, Rt = np.linalg.svd(M, full_matrices=False)
        D = rpca_utils.soft_thresholding(D, self.tau)
        M = (U * D) @ Rt @ V.T

        A = np.where(Omega, X - M, 0)

        return M, A

    # def fit(self, D: NDArray, y=None) -> SoftImpute:
    #     """Fit the imputer on D.

    #     Parameters
    #     ----------
    #     D : NDArray
    #         Input data

    #     y : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     self : object
    #         The fitted `SoftImpute` class instance.
    #     """
    #     D = D.copy()
    #     D = utils.prepare_data(D, self.period)

    #     if not isinstance(D, np.ndarray):
    #         raise AssertionError("Invalid type. D must be a NDArray.")

    #     n, m = D.shape
    #     mask = np.isnan(D)
    #     V = np.zeros((m, self.rank))
    #     U = self.random_state.normal(0.0, 1.0, (n, self.rank))
    #     U, _, _ = np.linalg.svd(U, full_matrices=False)
    #     Dsq = np.ones((self.rank, 1))
    #     col_means = np.nanmean(D, axis=0)
    #     np.copyto(D, col_means, where=np.isnan(D))
    #     if self.rank is None:
    #         self.rank = rpca_utils.approx_rank(D)
    #     for iter_ in range(self.max_iterations):
    #         U_old = U
    #         V_old = V
    #         Dsq_old = Dsq

    #         Q = U.T @ D
    #         if self.tau > 0:
    #             tmp = Dsq / (Dsq + self.tau)
    #             Q = Q * tmp
    #         Bsvd = np.linalg.svd(Q.T, full_matrices=False)
    #         V = Bsvd[0]
    #         Dsq = Bsvd[1][:, np.newaxis]
    #         U = U @ Bsvd[2]
    #         tmp = Dsq * V.T
    #         D_hat = U @ tmp
    #         D[mask] = D_hat[mask]

    #         L = (D @ V).T
    #         if self.tau > 0:
    #             tmp = Dsq / (Dsq + self.tau)
    #             L = L * tmp
    #         Lsvd = np.linalg.svd(L.T, full_matrices=False)
    #         U = Lsvd[0]
    #         Dsq = Lsvd[1][:, np.newaxis]
    #         V = V @ Lsvd[2]
    #         tmp = Dsq * V.T
    #         D_hat = U @ tmp
    #         D[mask] = D_hat[mask]

    #         ratio = self._check_convergence(U_old, Dsq_old, V_old, U, Dsq, V)
    #         if self.verbose:
    #             print(f"iter {iter_}: ratio = {round(ratio, 4)}")
    #         if ratio < self.tolerance:
    #             break

    #     self.u = U[:, : self.rank]
    #     self.d = Dsq[: self.rank]
    #     self.v = V[:, : self.rank]

    #     return self

    def _check_convergence(
        self,
        U_old: NDArray,
        D_old: NDArray,
        V_old: NDArray,
        U: NDArray,
        D: NDArray,
        V: NDArray,
    ) -> float:
        """Given a pair of iterates (U_old, D_old, V_old) and (U, D, V),
        it computes the relative change in Frobenius norm given by
        || U_old @  D_old^2 @ V_old.T - U @  D^2 @ V.T ||_F^2
        / || U_old @  D_old^2 @ V_old.T ||_F^2

        Parameters
        ----------
        U_old : NDArray
            previous matrix U
        D_old : NDArray
            previous matrix D
        V_old : NDArray
            previous matrix V
        U : NDArray
            current matrix U
        D : NDArray
            current matrix D
        V : NDArray
            current matrix V

        Returns
        -------
        float
            relative change
        """
        if any(arg is None for arg in (U_old, D_old, V_old, U, D, V)):
            raise ValueError("One or more arguments are None.")

        tr_D4 = (D**4).sum()
        tr_D_old4 = (D_old**4).sum()
        DUtU = D**2 * (U.T @ U_old)
        DVtV = D_old**2 * (V_old.T @ V)
        cross_term = (DUtU @ DVtV).diagonal().sum()
        return (tr_D_old4 + tr_D4 - 2 * cross_term) / max(tr_D_old4, 1e-9)

    # def transform(self, D: NDArray) -> NDArray:
    #     """Impute all missing values in D.

    #     Parameters
    #     ----------
    #     D : array-like of shape (n_samples, n_features)
    #         The input data to complete.

    #     Returns
    #     -------
    #     D : NDArray
    #         The imputed dataset.
    #     """
    #     D_transformed = self.u @ np.diag(self.d.T[0]) @ (self.v).T
    #     if self.projected:
    #         D_ = utils.prepare_data(D, self.period)
    #         mask = np.isnan(D_)
    #         D_transformed[~mask] = D_[~mask]

    #     D_transformed = utils.get_shape_original(D_transformed, D.shape)

    #     if np.all(np.isnan(D_transformed)):
    #         raise AssertionError("Result contains NaN. This is a bug.")

    #     return D_transformed
