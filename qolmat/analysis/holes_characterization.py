from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2

from qolmat.utils.exceptions import TooManyMissingPatterns
from qolmat.imputations.imputers import ImputerEM


class McarTest(ABC):
    """
    Astract class for MCAR tests.
    """

    @abstractmethod
    def test(self, df: pd.DataFrame) -> float:
        pass


class LittleTest(McarTest):
    """
    This class implements the Little's test, which is designed to detect the heterogeneity accross
    the missing patterns. The null hypothesis is "The missing data mechanism is MCAR". The
    shortcoming of this test is that it won't detect the heterogeneity of covariance.

    References
    ----------
    Little. "A Test of Missing Completely at Random for Multivariate Data with Missing Values."
    Journal of the American Statistical Association, Volume 83, 1988 - Issue 404

    Parameters
    ----------
    imputer : Optional[ImputerEM]
        Imputer based on the EM algorithm. The 'model' attribute must be equal to 'multinormal'.
        If None, the default ImputerEM is taken.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
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
        imputer = imputer._fit_element(df)

        d0 = 0
        n_rows, n_cols = df.shape
        degree_f = -n_cols
        ml_means = imputer.means
        ml_cov = n_rows / (n_rows - 1) * imputer.cov

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

        return 1 - float(chi2.cdf(d0, degree_f))


class PKLMTest(McarTest):
    """
    PKLMTest extends McarTest for testing purposes.
    
    Attributes:
    -----------
    nb_projections : int
        Number of projections.
    nb_permutation : int
        Number of permutations.
    nb_trees_per_proj : int
        Number of trees per projection.
    exact_p_value : bool
        If True, compute exact p-value.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        nb_projections: int = 100,
        nb_permutation: int = 30,
        nb_trees_per_proj: int = 200,
        exact_p_value: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__()
        self.nb_projections = nb_projections
        self.nb_permutation = nb_permutation
        self.nb_trees_per_proj = nb_trees_per_proj
        self.exact_p_value = exact_p_value
        self.random_state = (
            np.random.default_rng(random_state) if isinstance(
                random_state,
                (type(None), int)
            ) else random_state
        )

    @staticmethod
    def _check_nb_patterns(df: np.ndarray) -> None:
        """
        This method examines a NumPy array to identify distinct patterns of missing values (NaNs).
        If the number of unique patterns exceeds the number of rows in the array, it raises a
        `TooManyMissingPatterns` exception.
        This condition comes from the PKLM paper, please see the reference if needed.

        Parameters:
        df (np.ndarray): 2D array with NaNs as missing values.

        Raises:
        TooManyMissingPatterns: If unique missing patterns exceed the number of rows.
        """
        n_rows, _ = df.shape
        indicator_matrix = ~np.isnan(df)
        patterns = set(map(tuple, indicator_matrix))
        nb_patterns = len(patterns)
        if nb_patterns > n_rows:
            raise TooManyMissingPatterns()

    @staticmethod
    def _draw_features_and_target_indexes(df: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Randomly selects features and a target from the dataframe.

        Parameters:
        -----------
        df : np.ndarray
            The input dataframe.

        Returns:
        --------
        Tuple[np.ndarray, int]
            Indices of selected features and the target.
        """
        _, p = df.shape
        nb_features = np.random.randint(1, p)
        features_idx = np.random.choice(range(p), size=nb_features, replace=False)
        target_idx = np.random.choice(np.setdiff1d(np.arange(p), features_idx))
        return features_idx, target_idx

    @staticmethod
    def check_draw(df: np.ndarray, features_idx: np.ndarray, target_idx: int) -> np.bool_:
        """
        Checks if the drawn features and target are valid.
        # TODO : Need to develop.

        Parameters:
        -----------
        df : np.ndarray
            The input dataframe.
        features_idx : np.ndarray
            Indices of the selected features.
        target_idx : int
            Index of the target.

        Returns:
        --------
        bool
            True if the draw is valid, False otherwise.
        """
        target_values = df[~np.isnan(df[:,features_idx]).any(axis=1)][:, target_idx]
        is_nan = np.isnan(target_values).any()
        is_distinct_values = (~np.isnan(target_values)).any()
        return is_nan and is_distinct_values

    def draw_projection(self, df: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Draws a valid projection of features and a target.

        Parameters:
        -----------
        df : np.ndarray
            The input dataframe.

        Returns:
        --------
        Tuple[np.ndarray, int]
            Indices of selected features and the target.
        """
        is_checked = False
        while not is_checked:
            features_idx, target_idx = self._draw_features_and_target_indexes(df)
            is_checked = self.check_draw(df, features_idx, target_idx)
        return features_idx, target_idx

    @staticmethod
    def _build_dataset(
        df: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = df[~np.isnan(df[:, features_idx]).any(axis=1)][:, features_idx]
        y = np.where(np.isnan(df[~np.isnan(df[:, features_idx]).any(axis=1)][:, target_idx]), 1, 0)
        return X, y

    @staticmethod
    def _build_label(
        df: np.ndarray,
        perm: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int
    ) -> np.ndarray:
        return perm[~np.isnan(df[:, features_idx]).any(axis=1), target_idx]


    def _get_oob_probabilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        clf = RandomForestClassifier(
            n_estimators=self.nb_trees_per_proj,
            #max_features=None,
            min_samples_split=10,
            bootstrap=True,
            oob_score=True,
        )
        clf.fit(X, y)
        return clf.oob_decision_function_

    @staticmethod
    def _U_hat(oob_probabilities: np.ndarray, labels: np.ndarray) -> float:
        oob_probabilities = np.clip(oob_probabilities, 1e-9, 1-1e-9)

        unique_labels = np.unique(labels)
        label_matrix = (labels[:, None] == unique_labels).astype(int)
        p_true = oob_probabilities * label_matrix
        p_false = oob_probabilities * (1 - label_matrix)

        p0_0 = p_true[:, 0][np.where(p_true[:, 0] != 0.)]
        p0_1 = p_false[:, 0][np.where(p_false[:, 0] != 0.)]
        p1_1 = p_true[:, 1][np.where(p_true[:, 1] != 0.)]
        p1_0 = p_false[:, 1][np.where(p_false[:, 1] != 0.)]

        if unique_labels.shape[0] == 1:
            if unique_labels[0] == 0:
                n0 = labels.shape[0]
                return np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p1_0 / (1 - p1_0)).sum() / n0
            else:
                n1 = labels.shape[0]
                return np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p0_1 / (1 - p0_1)).sum() / n1

        n0, n1 = label_matrix.sum(axis=0)
        u_0 = np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p0_1 / (1 - p0_1)).sum() / n1
        u_1 = np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p1_0 / (1 - p1_0)).sum() / n0

        return u_0 + u_1

    def test(self, df: np.ndarray):
        self._check_nb_patterns(df)
