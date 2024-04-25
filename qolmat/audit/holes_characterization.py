from __future__ import annotations
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2

from qolmat.imputations.imputers import ImputerEM

if TYPE_CHECKING:
    from qolmat.imputations.imputers import _Imputer


class MCARTest:
    """
    This class implements the statistical tests to test the MCAR case.

    Parameters
        ----------
        method : Literal[&quot;little&quot;]
            The name of the statistical test. This should be handled by qolmat.
        imputer : Optional[_Imputer], optional
            If the selected test needs a imputer, you can provide the Imputer you want. Otherwise,
            a default imputer will be used.
    """

    def __init__(self, method: Literal["little"], imputer: Optional[_Imputer] = None):
        if method not in ["little"]:
            raise ValueError(f"method` must be handled by qolmat, provided value is '{method}'")

        self.method = method
        self.imputer = imputer

    def test(self, df: pd.DataFrame) -> float:
        if self.method == "little":
            return self.little_mcar_test(df)

    def little_mcar_test(self, df: pd.DataFrame) -> float:
        """
        This method implements the Little's test. Use this test to test the homogenity of means
        between all your missing patterns.
        The null hypethoses is "The missing data mechanism is MCAR".
        Be aware that this test won't detect the heterogeneity of covariance.

        Parameters
        ----------
        df : pd.DataFrame
            Your input data with missing values.

        Returns
        -------
        float
            The p-value of the test.
        """
        imputer = self.imputer or ImputerEM()
        fitted_imputer = imputer._fit_element(df)

        # Instanciant the stat, the degree of freedom and estimators.
        d0 = 0
        n_rows, degree_f = df.shape
        degree_f = -degree_f
        ml_means = fitted_imputer.means
        ml_cov = n_rows / (n_rows - 1) * fitted_imputer.cov

        # Iterate over the patterns
        df_nan = df.notna()
        for tup_pattern, df_nan_pattern in df_nan.groupby(df_nan.columns.tolist()):
            n_rows_pattern, _ = df_nan_pattern.shape
            ind_pattern = df_nan_pattern.index
            df_pattern = df.loc[ind_pattern, list(tup_pattern)]
            obs_mean = df_pattern.mean().to_numpy()

            diff_means = obs_mean - ml_means[list(tup_pattern)]
            inv_sigma_pattern = np.linalg.inv(ml_cov[:, tup_pattern][tup_pattern, :])

            d0 += n_rows_pattern * np.dot(np.dot(diff_means, inv_sigma_pattern), diff_means.T)
            degree_f += tup_pattern.count(True)

        return 1 - chi2.cdf(d0, degree_f)
