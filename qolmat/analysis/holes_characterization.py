"""Script for characterising the holes."""

from abc import ABC, abstractmethod
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders.one_hot import OneHotEncoder
from joblib import Parallel, delayed
from scipy.stats import chi2
from sklearn import utils as sku
from sklearn.ensemble import RandomForestClassifier

from qolmat.imputations.imputers import ImputerEM
from qolmat.utils.input_check import check_pd_df_dtypes
from qolmat.utils.utils import RandomSetting


class McarTest(ABC):
    """Abstract class for MCAR tests.

    Parameters
    ----------
    random_state : int or np.random.RandomState, optional
        Seed or random state for reproducibility.

    Methods
    -------
    test
        Abstract method to perform the MCAR test on the given DataFrame or
        NumPy array.

    """

    def __init__(self, random_state: RandomSetting = None):
        """Initialize the McarTest class with a random state.

        Parameters
        ----------
        random_state : int or np.random.RandomState, optional
            Seed or random state for reproducibility.

        """
        self.rng = sku.check_random_state(random_state)

    @abstractmethod
    def test(
        self, df: Union[pd.DataFrame, np.ndarray]
    ) -> Union[float, Tuple[float, List[float]]]:
        """Perform the MCAR test on the input data.

        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            Data to be tested for MCAR. Can be provided as a pandas DataFrame
            or a NumPy array.

        Returns
        -------
        float or tuple of float and list of float
            Test statistic, or a tuple with the test statistic and additional
            details if applicable.

        """
        raise NotImplementedError


class LittleTest(McarTest):
    """Little Test class.

    This class implements the Little's test, which is designed to detect the
    heterogeneity accross the missing patterns. The null hypothesis is
    "The missing data mechanism is MCAR". The shortcoming of this test is
    that it won't detect the heterogeneity of covariance.

    References
    ----------
    Little. "A Test of Missing Completely at Random for Multivariate Data with
    Missing Values." Journal of the American Statistical Association,
    Volume 83, 1988 - Issue 404

    Parameters
    ----------
    imputer : Optional[ImputerEM]
        Imputer based on the EM algorithm. The 'model' attribute must be
        equal to 'multinormal'. If None, the default ImputerEM is taken.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.

    """

    def __init__(
        self,
        imputer: Optional[ImputerEM] = None,
        random_state: RandomSetting = None,
    ):
        super().__init__()
        if imputer and imputer.model != "multinormal":
            raise AttributeError(
                "The ImputerEM model must be 'multinormal' "
                "to use the Little's test"
            )
        self.imputer = imputer
        self.random_state = random_state

    def test(self, df: pd.DataFrame) -> float:
        """Apply the Little's test over a real dataframe.

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
        for tup_pattern, df_nan_pattern in df_nan.groupby(
            df_nan.columns.tolist()
        ):
            n_rows_pattern, _ = df_nan_pattern.shape
            ind_pattern = df_nan_pattern.index
            df_pattern = df.loc[ind_pattern, list(tup_pattern)]
            obs_mean = df_pattern.mean().to_numpy()

            diff_means = obs_mean - ml_means[list(tup_pattern)]
            inv_sigma_pattern = np.linalg.inv(
                ml_cov[:, tup_pattern][tup_pattern, :]
            )

            d0 += n_rows_pattern * np.dot(
                np.dot(diff_means, inv_sigma_pattern), diff_means.T
            )
            degree_f += tup_pattern.count(True)

        return 1 - float(chi2.cdf(d0, degree_f))


class PKLMTest(McarTest):
    """PKLM Test class.

    This class implements the PKLM test, a fully non-parametric, easy-to-use
    and powerful test for the missing completely at random (MCAR) assumption on
    the missingness mechanism of a dataset.
    The null hypothesis is "The missing data mechanism is MCAR".

    This test is applicable to mixed data (quantitative and categoricals) types


    If you're familiar with the paper, this implementation of the PKLM test was
    made for the parameter size.resp.set=2 only.

    References
    ----------
    Spohn, M. L., Näf, J., Michel, L., & Meinshausen, N. (2021). PKLM: A
    flexible MCAR test using Classification. arXiv preprint arXiv:2109.10150.

    Parameters
    ----------
    nb_projections : int
        Number of projections.
    nb_projections_threshold : int
        If the maximum number of possible permutations is less than this
        threshold, then all projections are used. Otherwise, nb_projections
        random projections are drawn.
    nb_permutation : int
        Number of permutations.
    nb_trees_per_proj : int
        Number of trees per projection.
    compute_partial_p_values : bool
        If true, compute the partial p-values.
    encoder : OneHotEncoder or None, default=None
        Encoder to convert non numeric pandas dataframe values to numeric
        values.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.

    """

    def __init__(
        self,
        nb_projections: int = 100,
        nb_projections_threshold: int = 200,
        nb_permutation: int = 30,
        nb_trees_per_proj: int = 200,
        compute_partial_p_values: bool = False,
        encoder: Union[None, OneHotEncoder] = None,
        random_state: RandomSetting = None,
    ):
        super().__init__(random_state=random_state)
        self.nb_projections = nb_projections
        self.nb_projections_threshold = nb_projections_threshold
        self.nb_permutation = nb_permutation
        self.nb_trees_per_proj = nb_trees_per_proj
        self.compute_partial_p_values = compute_partial_p_values
        self.encoder = encoder

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Encode the DataFrame.

        Encode the DataFrame by converting numeric columns to a numpy array
        and applying one-hot encoding to objects and boolean columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be encoded.

        Returns
        -------
        np.ndarray
            The encoded DataFrame as a numpy ndarray, with numeric data
            concatenated with one-hot encoded categorical and boolean data.

        """
        if not df.select_dtypes(include=["object", "bool"]).columns.to_list():
            return df.to_numpy()

        if not self.encoder:
            self.encoder = OneHotEncoder(
                cols=df.select_dtypes(include=["object", "bool"]).columns,
                return_df=False,
                handle_missing="return_nan",
            )

        return self.encoder.fit_transform(df)

    def _pklm_preprocessing(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Preprocess the input DataFrame or ndarray for further processing.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input data to be preprocessed. Can be a pandas DataFrame or a
            numpy ndarray.

        Returns
        -------
        np.ndarray
            The preprocessed data as a numpy ndarray.

        Raises
        ------
        TypeNotHandled
            If the DataFrame contains columns with data types that are not
            numeric, string, or boolean.

        """
        if isinstance(X, np.ndarray):
            return X

        check_pd_df_dtypes(
            X,
            [
                pd.api.types.is_numeric_dtype,
                pd.api.types.is_string_dtype,
                pd.api.types.is_bool_dtype,
            ],
        )
        return self._encode_dataframe(X)

    @staticmethod
    def _get_max_draw(p: int) -> int:
        """Calculate the number of possible projections.

        Parameters
        ----------
        p : int
            The number of columns of the input matrix.

        Returns
        -------
        int
            The number of possible projections.

        """
        return p * (2 ** (p - 1) - 1)

    def _draw_features_and_target_indexes(
        self, X: np.ndarray
    ) -> Tuple[List[int], int]:
        """Randomly select features and a target from the dataframe.

        This corresponds to the Ai and Bi projections of the paper.

        Parameters
        ----------
        X : np.ndarray
            The input dataframe.

        Returns
        -------
        Tuple[np.ndarray, int]
            Indices of selected features and the target.

        """
        _, p = X.shape
        nb_features = self.rng.randint(1, p)
        features_idx = self.rng.choice(p, size=nb_features, replace=False)
        target_idx = self.rng.choice(np.setdiff1d(np.arange(p), features_idx))
        return features_idx.tolist(), target_idx

    @staticmethod
    def _check_draw(
        X: np.ndarray, features_idx: List[int], target_idx: int
    ) -> bool:
        """Check if the drawn features and target are valid.

        Here we check
        that the number of induced classes is equal to 2. Using the notation
        from the paper, we want |G(Ai, Bi)| = 2.

        Parameters
        ----------
        X : np.ndarray
            The input dataframe.
        features_idx : np.ndarray
            Indices of the selected features.
        target_idx : int
            Index of the target.

        Returns
        -------
        bool
            True if the draw is valid, False otherwise.

        """
        target_values = X[~np.isnan(X[:, features_idx]).any(axis=1)][
            :, target_idx
        ]
        is_nan = np.isnan(target_values).any()
        is_distinct_values = (~np.isnan(target_values)).any()
        return is_nan and is_distinct_values

    def _generate_label_feature_combinations(
        self, X: np.ndarray
    ) -> List[Tuple[List[int], int]]:
        """Generate all valid combinations of features and labels.

        Parameters
        ----------
        X : np.ndarray
            The input data array.

        Returns
        -------
        List[Tuple[int, List[int]]]
            A list of tuples where each tuple contains a label and a list of
            selected features that can be used for projection.

        """
        _, p = X.shape
        indices = list(range(p))
        result = []

        for label in indices:
            feature_candidates = [i for i in indices if i != label]

            for r in range(1, len(feature_candidates) + 1):
                for feature_set in combinations(feature_candidates, r):
                    if self._check_draw(X, list(feature_set), label):
                        result.append((list(feature_set), label))

        return result

    def _draw_projection(self, X: np.ndarray) -> Tuple[List[int], int]:
        """Draw a valid projection of features and a target.

        If nb_projections_threshold < _get_max_draw(X.shape[1]).

        Parameters
        ----------
        X : np.ndarray
            The input dataframe.

        Returns
        -------
        Tuple[np.ndarray, int]
            Indices of selected features and the target.

        """
        is_checked = False
        while not is_checked:
            features_idx, target_idx = self._draw_features_and_target_indexes(
                X
            )
            is_checked = self._check_draw(X, features_idx, target_idx)
        return features_idx, target_idx

    @staticmethod
    def _build_dataset(
        X: np.ndarray, features_idx: np.ndarray, target_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a dataset for a classification task.

        Build a dataset by selecting specified features and target from a
        NumPy array, excluding rows with NaN values in the feature columns.
        For the label, we create a binary classification problem where yi =1 if
        target_idx_i is missing.

        Parameters
        ----------
        X: np.ndarray
            Input data array.
        features_idx: np.ndarray
            Indices of the feature columns.
        target_idx: int
            Index of the target column.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Full observed array of selected features.
            - y (np.ndarray): Binary array indicating presence of NaN (1) in
            the target column.

        """
        X_features = X[~np.isnan(X[:, features_idx]).any(axis=1)][
            :, features_idx
        ]
        y = np.where(
            np.isnan(
                X[~np.isnan(X[:, features_idx]).any(axis=1)][:, target_idx]
            ),
            1,
            0,
        )
        return X_features, y

    @staticmethod
    def _build_label(
        X: np.ndarray,
        perm: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int,
    ) -> np.ndarray:
        """Build a label.

        Build a label array by selecting target values from a permutation
        array, excluding rows with NaN values in the specified feature columns.
        For the label, we create a binary classification problem where yi =1 if
        target_idx_i is missing.

        Parameters
        ----------
        X: np.ndarray
            Input data array.
        perm: np.ndarray
            Permutation array from which labels are selected.
        features_idx: np.ndarray
            Indices of the feature columns.
        target_idx: int
            Index of the target column in the permutation array.

        Returns
        -------
            np.ndarray: Binary array indicating presence of NaN (1) in the
            target column.

        """
        return perm[~np.isnan(X[:, features_idx]).any(axis=1), target_idx]

    def _get_oob_probabilities(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Retrieve out-of-bag probabilities.

        Train a RandomForestClassifier and retrieves out-of-bag (OOB)
        probabilities.

        Parameters
        ----------
        X: np.ndarray
            Feature array for training.
        y: np.ndarray
            Target array for training.

        Returns
        -------
            np.ndarray: Out-of-bag probabilities for each class.

        """
        clf = RandomForestClassifier(
            n_estimators=self.nb_trees_per_proj,
            min_samples_split=10,
            bootstrap=True,
            oob_score=True,
            random_state=self.rng,
            max_features=1.0,
        )
        clf.fit(X, y)
        return clf.oob_decision_function_

    @staticmethod
    def _U_hat(oob_probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute the U_hat statistic.

        U_hat is a measure of classifier performance, using out-of-bag
        probabilities and true labels.

        Parameters
        ----------
        oob_probabilities: np.ndarray
            Out-of-bag probabilities for each class.
        labels: np.ndarray
            True labels for the data.

        Returns
        -------
            float: The computed U_hat statistic.

        """
        if oob_probabilities.shape[1] == 1:
            return 0.0

        oob_probabilities = np.clip(oob_probabilities, 1e-9, 1 - 1e-9)

        unique_labels = np.unique(labels)
        label_matrix = (labels[:, None] == unique_labels).astype(int)
        p_true = oob_probabilities * label_matrix
        p_false = oob_probabilities * (1 - label_matrix)

        p0_0 = p_true[:, 0][np.where(p_true[:, 0] != 0.0)]
        p0_1 = p_false[:, 0][np.where(p_false[:, 0] != 0.0)]
        p1_1 = p_true[:, 1][np.where(p_true[:, 1] != 0.0)]
        p1_0 = p_false[:, 1][np.where(p_false[:, 1] != 0.0)]

        if unique_labels.shape[0] == 1:
            if unique_labels[0] == 0:
                n0 = labels.shape[0]
                return (
                    np.log(p0_0 / (1 - p0_0)).sum() / n0
                    - np.log(p1_0 / (1 - p1_0)).sum() / n0
                )
            else:
                n1 = labels.shape[0]
                return (
                    np.log(p1_1 / (1 - p1_1)).sum() / n1
                    - np.log(p0_1 / (1 - p0_1)).sum() / n1
                )

        n0, n1 = label_matrix.sum(axis=0)
        u_0 = (
            np.log(p0_0 / (1 - p0_0)).sum() / n0
            - np.log(p0_1 / (1 - p0_1)).sum() / n1
        )
        u_1 = (
            np.log(p1_1 / (1 - p1_1)).sum() / n1
            - np.log(p1_0 / (1 - p1_0)).sum() / n0
        )

        return u_0 + u_1

    def _parallel_process_permutation(
        self,
        X: np.ndarray,
        M_perm: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int,
        oob_probabilities: np.ndarray,
    ) -> float:
        """Process a permutation.

        Parameters
        ----------
        X : np.ndarray
            input array
        M_perm : np.ndarray
            permutation array
        features_idx : np.ndarray
            index of the features
        target_idx : int
            index of the target
        oob_probabilities : np.ndarray
            out of bag probabilities

        Returns
        -------
        float
            esimtated statistic U_hat

        """
        y = self._build_label(X, M_perm, features_idx, target_idx)
        return self._U_hat(oob_probabilities, y)

    def _parallel_process_projection(
        self,
        X: np.ndarray,
        list_permutations: List[np.ndarray],
        features_idx: np.ndarray,
        target_idx: int,
    ) -> Tuple[float, List[float]]:
        """Compute statistics for a projection.

        Parameters
        ----------
        X : np.ndarray
            input array
        list_permutations : List[np.ndarray]
            list of permutations
        features_idx : np.ndarray
            index of the features
        target_idx : int
            index of the target

        Returns
        -------
        Tuple[float, List[float]]
            estimated statistic u_hat and list of u_hat for each permutation

        """
        X_features, y = self._build_dataset(X, features_idx, target_idx)
        oob_probabilities = self._get_oob_probabilities(X_features, y)
        u_hat = self._U_hat(oob_probabilities, y)
        # We iterate over the permutation because for a given projection
        # We fit only one classifier to get oob probabilities and compute u_hat
        # nb_permutations times.
        result_u_permutations = Parallel(n_jobs=-1)(
            delayed(self._parallel_process_permutation)(
                X, M_perm, features_idx, target_idx, oob_probabilities
            )
            for M_perm in list_permutations
        )
        return u_hat, result_u_permutations

    @staticmethod
    def _build_B(list_proj: List, n_cols: int) -> np.ndarray:
        """Construct a binary matrix B based on the given projections.

        Parameters
        ----------
        list_proj : List
            A list of tuples where each tuple represents a projection, and the
            second element of each tuple is an index used to build the target.
        n_cols : int
            The number of columns in the resulting matrix B.

        Returns
        -------
        np.ndarray
            A binary matrix of shape (n_cols, len(list_proj)) where each column
            corresponds to a projection, and the entries are 0 or 1 based on
            the projections.

        """
        list_bi = [projection[1] for projection in list_proj]
        B = np.ones((len(list_proj), n_cols), dtype=int)

        for j in range(len(list_bi)):
            B[j, list_bi[j]] = 0

        return B.transpose()

    def _compute_partial_p_value(
        self, B: np.ndarray, U: np.ndarray, U_sigma: np.ndarray, k: int
    ) -> float:
        """Compute the partial p-values.

        Compute the partial p-value for a statistical test based on a given
        permutation.

        Parameters
        ----------
        B : np.ndarray
            Pass matrix indicating the column used to create the target in each
            projection.
        U : np.ndarray
            A vector of shape (nb_permutations,) representing the observed test
            statistics.
        U_sigma : np.ndarray
            A matrix of shape (nb_permutations, nb_observations) where each row
            represents the test statistics for a given projection and all the
            permutations.
        k : int
            The index of the column on which to compute the partial p_value.

        Returns
        -------
        float
            The partial p-value.

        """
        U_k = B[k, :] @ U
        p_v_k = 1.0

        for u_sigma_k in (B[k, :] @ U_sigma).tolist():
            if u_sigma_k >= U_k:
                p_v_k += 1

        return p_v_k / (self.nb_permutation + 1)

    def test(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[float, Tuple[float, List[float]]]:
        """Apply the PKLM test over a real dataset.

        Parameters
        ----------
        X : np.ndarray
            The input dataset with missing values.

        Returns
        -------
        float
            If compute_partial_p_values=False. Returns the p-value of the test.
        Tuple[float, List[float]]
            If compute_partial_p_values=True. Returns the p-value of the test
            and the list of all the partial p-values.

        """
        X = self._pklm_preprocessing(X)
        _, n_cols = X.shape

        if self._get_max_draw(n_cols) <= self.nb_projections_threshold:
            list_proj = self._generate_label_feature_combinations(X)
        else:
            list_proj = [
                self._draw_projection(X) for _ in range(self.nb_projections)
            ]

        M = np.isnan(X).astype(int)
        list_perm = [
            self.rng.permutation(M) for _ in range(self.nb_permutation)
        ]
        U = 0.0
        list_U_sigma = [0.0 for _ in range(self.nb_permutation)]

        parallel_results = Parallel(n_jobs=-1)(
            delayed(self._parallel_process_projection)(
                X, list_perm, features_idx, target_idx
            )
            for features_idx, target_idx in list_proj
        )

        for U_projection, results in parallel_results:
            U += U_projection
            list_U_sigma = [x + y for x, y in zip(list_U_sigma, results)]

        U = U / self.nb_projections
        list_U_sigma = [x / self.nb_permutation for x in list_U_sigma]

        p_value = 1.0
        for u_sigma in list_U_sigma:
            if u_sigma >= U:
                p_value += 1

        p_value = p_value / (self.nb_permutation + 1)

        if not self.compute_partial_p_values:
            return p_value
        else:
            B = self._build_B(list_proj, n_cols)
            U_matrix = np.array(
                [np.atleast_1d(item[0]) for item in parallel_results]
            )
            U_sigma = np.array(
                [np.atleast_1d(item[1]) for item in parallel_results]
            )
            p_values = [
                self._compute_partial_p_value(B, U_matrix, U_sigma, k)
                for k in range(n_cols)
            ]
            return p_value, p_values
