from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from qolmat.imputations.imputers import ImputerEM


class MCARTest(ABC):
    """
    Astract class for MCAR tests.
    """

    @abstractmethod
    def test(self, df: pd.DataFrame) -> float:
        pass


class LittleTest(MCARTest):
    """
    This class implements the Little's test. The Little's test is designed to detect the
    heterogeneity accross the missing patterns. The null hypothesis is "The missing data mechanism
    is MCAR". Be aware that this test won't detect the heterogeneity of covariance.

    References
    ----------
    Little. "A Test of Missing Completely at Random for Multivariate Data with Missing Values."
    Journal of the American Statistical Association, Volume 83, 1988 - Issue 404

    Parameters
    ----------
    imputer : Optional[ImputerEM]
        Imputer based on the EM algorithm. The 'model' attribute must be equal to 'multinormal'.
        If None, the default ImputerEM is taken.
    random_state : Union[None, int, np.random.RandomState], optional
        Controls the randomness of the fit_transform, by default None
    """

    def __init__(
        self,
        imputer: Optional[ImputerEM] = None,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__()
        if imputer and imputer.model != "multinormal":
            raise AttributeError(
                "The ImputerEM model must be 'multinormal' to use the Little's test"
            )
        self.imputer = imputer
        self.random_state = random_state

    def test(self, df: pd.DataFrame) -> float:
        """
        Apply the Little's test over a real dataframe.


        Parameters
        ----------
        df : pd.DataFrame
            The input dataset with missing values.

        Returns
        -------
        float
            The p-value of the test.
        """
        imputer = self.imputer or ImputerEM(random_state=self.random_state)
        fitted_imputer = imputer._fit_element(df)

        d0 = 0
        n_rows, n_cols = df.shape
        degree_f = -n_cols
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
