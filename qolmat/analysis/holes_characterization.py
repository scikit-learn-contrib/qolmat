from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import utils as sku
from scipy.stats import chi2

from qolmat.utils.exceptions import TooManyMissingPatterns, TypeNotHandled
from qolmat.imputations.imputers import ImputerEM


class McarTest(ABC):
    """
    Astract class for MCAR tests.

    Parameters
    ----------
    random_state : int, optional
        The seed of the pseudo random number generator to use, for reproductibility.
    """

    def __init__(self, random_state: Union[None, int, np.random.RandomState] = None):
        self.rng = sku.check_random_state(random_state)

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
    Journal of the American Statistical Association, Volume 83, 1988 - no 404, p. 1198-1202

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
        super().__init__(random_state=random_state)
        if imputer and imputer.model != "multinormal":
            raise AttributeError(
                "The ImputerEM model must be 'multinormal' to use the Little's test"
            )
        self.imputer = imputer

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
        imputer = self.imputer or ImputerEM(random_state=self.rng)
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

            d0 += n_rows_pattern * np.dot(
                np.dot(diff_means, inv_sigma_pattern), diff_means.T
            )
            degree_f += tup_pattern.count(True)

        return 1 - float(chi2.cdf(d0, degree_f))


class PKLMTest(McarTest):
    """
    This class implements the PKLM test, a fully non-parametric, easy-to-use, and powerful test
    for the missing completely at random (MCAR) assumption on the missingness mechanism of a
    dataset. The null hypothesis is "The missing data mechanism is MCAR".

    This test is applicable to mixed data (quantitative and categoricals) types.

    References
    ----------
    Spohn, M. L., Näf, J., Michel, L., & Meinshausen, N. (2021). PKLM: A flexible MCAR test using
    Classification. arXiv preprint arXiv:2109.10150.

    Parameters
    -----------
    nb_projections : int
        Number of projections.
    nb_permutation : int
        Number of permutations.
    nb_trees_per_proj : int
        Number of trees per projection.
    compute_partial_p_values : bool
        If true, compute the partial p-values.
    exact_p_value : bool
        If True, compute exact p-value.
    encoder : OneHotEncoder or None, default=None
        Encoder to convert non numeric pandas dataframe values to numeric values.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        nb_projections: int = 100,
        nb_permutation: int = 30,
        nb_trees_per_proj: int = 200,
        compute_partial_p_values: bool = False,
        exact_p_value: bool = False,
        encoder: Union[None, OneHotEncoder] = None,  # We could define more encoders.
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__(random_state=random_state)
        self.nb_projections = nb_projections
        self.nb_permutation = nb_permutation
        self.nb_trees_per_proj = nb_trees_per_proj
        self.compute_partial_p_values = compute_partial_p_values
        self.exact_p_value = exact_p_value
        self.encoder = encoder

        if self.exact_p_value:
            self.process_permutation = self._parallel_process_permutation_exact
        self.process_permutation = self._parallel_process_permutation

    @staticmethod
    def _check_nb_patterns(df: np.ndarray) -> None:
        """
        This method examines a NumPy array to identify distinct patterns of missing values (NaNs).
        If the number of unique patterns exceeds the number of rows in the array, it raises a
        `TooManyMissingPatterns` exception.
        This condition comes from the PKLM paper, please see the reference if needed.

        Parameters:
        -----------
        df : np.ndarray
            2D array with NaNs as missing values.

        Raises:
        -------
            TooManyMissingPatterns: If unique missing patterns exceed the number of rows.
        """
        n_rows, _ = df.shape
        indicator_matrix = ~np.isnan(df)
        patterns = set(map(tuple, indicator_matrix))
        nb_patterns = len(patterns)
        if nb_patterns > n_rows:
            raise TooManyMissingPatterns()

    @staticmethod
    def _check_pd_df_dtypes(df: pd.DataFrame):
        """
        Validates that the columns of the DataFrame have allowed data types.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame whose columns' data types are to be checked.

        Raises:
        -------
        TypeNotHandled
            If any column has a data type that is not numeric, string, or boolean.
        """
        allowed_types = [
            pd.api.types.is_numeric_dtype,
            pd.api.types.is_string_dtype,
            pd.api.types.is_bool_dtype,
        ]

        def is_allowed_type(dtype):
            return any(check(dtype) for check in allowed_types)

        invalid_columns = [
            (col, dtype)
            for col, dtype in df.dtypes.items()
            if not is_allowed_type(dtype)
        ]
        if invalid_columns:
            for column_name, dtype in invalid_columns:
                raise TypeNotHandled(col=column_name, type_col=dtype)

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encodes the DataFrame by converting numeric columns to a numpy array
        and applying one-hot encoding to non-numeric columns.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to be encoded.

        Returns:
        -------
        np.ndarray
            The encoded DataFrame as a numpy ndarray, with numeric data concatenated
            with one-hot encoded categorical and boolean data.
        """
        df_numerics = df.select_dtypes(include=["number"]).to_numpy()

        if not self.encoder:
            self.encoder = OneHotEncoder()

        df_non_numerics = self.encoder.fit_transform(
            df.select_dtypes(include=["object", "bool"])
        ).toarray()

        return np.concatenate((df_numerics, df_non_numerics), axis=1)

    def _pklm_preprocessing(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocesses the input DataFrame or ndarray for further processing.

        Parameters:
        -----------
        df : Union[pd.DataFrame, np.ndarray]
            The input data to be preprocessed. Can be a pandas DataFrame or a numpy ndarray.

        Returns:
        -------
        np.ndarray
            The preprocessed data as a numpy ndarray.

        Raises:
        -------
        TypeNotHandled
            If the DataFrame contains columns with data types that are not numeric, string, or boolean.
        """
        if isinstance(df, np.ndarray):
            return df

        self._check_pd_df_dtypes(df)
        return self._encode_dataframe(df)

    def _draw_features_and_target_indexes(self, df: np.ndarray) -> Tuple[np.ndarray, int]:
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
        nb_features = self.rng.randint(1, p)
        features_idx = self.rng.choice(range(p), size=nb_features, replace=False)
        target_idx = self.rng.choice(np.setdiff1d(np.arange(p), features_idx))
        return features_idx, target_idx

    @staticmethod
    def _check_draw(df: np.ndarray, features_idx: np.ndarray, target_idx: int) -> np.bool_:
        """
        Checks if the drawn features and target are valid.
        # TODO : Need to develop ?

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
        target_values = df[~np.isnan(df[:, features_idx]).any(axis=1)][:, target_idx]
        is_nan = np.isnan(target_values).any()
        is_distinct_values = (~np.isnan(target_values)).any()
        return is_nan and is_distinct_values

    def _draw_projection(self, df: np.ndarray) -> Tuple[np.ndarray, int]:
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
            is_checked = self._check_draw(df, features_idx, target_idx)
        return features_idx, target_idx

    @staticmethod
    def _build_dataset(
        df: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a dataset by selecting specified features and target from a NumPy array,
        excluding rows with NaN values in the feature columns.

        Parameters:
        -----------
        df: np.ndarray
            Input data array.
        features_idx: np.ndarray
            Indices of the feature columns.
        target_idx: int
            Index of the target column.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Array of selected features.
            - y (np.ndarray): Binary array indicating presence of NaN (1) in the target column.
        """
        X = df[~np.isnan(df[:, features_idx]).any(axis=1)][:, features_idx]
        y = np.where(
            np.isnan(df[~np.isnan(df[:, features_idx]).any(axis=1)][:, target_idx]),
            1,
            0,
        )
        return X, y

    @staticmethod
    def _build_label(
        df: np.ndarray,
        perm: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int
    ) -> np.ndarray:
        """
        Builds a label array by selecting target values from a permutation array,
        excluding rows with NaN values in the specified feature columns.

        Parameters:
        -----------
        df: np.ndarray
            Input data array.
        perm: np.ndarray
            Permutation array from which labels are selected.
        features_idx: np.ndarray
            Indices of the feature columns.
        target_idx: int
            Index of the target column in the permutation array.

        Returns:
        --------
            np.ndarray: Binary array indicating presence of NaN (1) in the target column.
        """
        return perm[~np.isnan(df[:, features_idx]).any(axis=1), target_idx]

    def _get_oob_probabilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Trains a RandomForestClassifier and retrieves out-of-bag (OOB) probabilities.

        Parameters:
        -----------
        X: np.ndarray
            Feature array for training.
        y: np.ndarray
            Target array for training.

        Returns:
        --------
            np.ndarray: Out-of-bag probabilities for each class.
        """
        clf = RandomForestClassifier(
            n_estimators=self.nb_trees_per_proj,
            # max_features=None,
            min_samples_split=10,
            bootstrap=True,
            oob_score=True,
            random_state=self.rng,
        )
        clf.fit(X, y)
        return clf.oob_decision_function_

    @staticmethod
    def _U_hat(oob_probabilities: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes the U_hat statistic, a measure of classifier performance, using
        out-of-bag probabilities and true labels.

        Parameters:
        -----------
        oob_probabilities: np.ndarray
            Out-of-bag probabilities for each class.
        labels: np.ndarray
            True labels for the data.

        Returns:
        --------
            float: The computed U_hat statistic.
        """
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
            np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p0_1 / (1 - p0_1)).sum() / n1
        )
        u_1 = (
            np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p1_0 / (1 - p1_0)).sum() / n0
        )

        return u_0 + u_1

    def _parallel_process_permutation(
        self,
        df: np.ndarray,
        M_perm: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int,
        oob_probabilities: np.ndarray,
    ) -> float:
        y = self._build_label(df, M_perm, features_idx, target_idx)
        return self._U_hat(oob_probabilities, y)

    def _parallel_process_permutation_exact(
        self,
        df: np.ndarray,
        M_perm: np.ndarray,
        features_idx: np.ndarray,
        target_idx: int,
        oob_probabilites_unused: np.ndarray,
    ) -> float:
        X, _ = self._build_dataset(df, features_idx, target_idx)
        y = self._build_label(df, M_perm, features_idx, target_idx)
        oob_probabilities = self._get_oob_probabilities(X, y)
        return self._U_hat(oob_probabilities, y)

    def _parallel_process_projection(
        self,
        df: np.ndarray,
        list_permutations: List[np.ndarray],
        features_idx: np.ndarray,
        target_idx: int,
    ) -> Tuple[float, List[float]]:
        X, y = self._build_dataset(df, features_idx, target_idx)
        oob_probabilities = self._get_oob_probabilities(X, y)
        u_hat = self._U_hat(oob_probabilities, y)
        result_u_permutations = Parallel(n_jobs=-1)(
            delayed(self.process_permutation)(
                df, M_perm, features_idx, target_idx, oob_probabilities
            )
            for M_perm in list_permutations
        )
        return u_hat, result_u_permutations

    @staticmethod
    def _build_B(list_proj: List, n_cols: int) -> np.ndarray:
        """
        Constructs a binary matrix B based on the given projections.

        Parameters:
        -----------
        list_proj : List
            A list of tuples where each tuple represents a projection, and the 
            second element of each tuple is an index used to build the target.
        n_cols : int
            The number of columns in the resulting matrix B.

        Returns:
        --------
        np.ndarray
            A binary matrix of shape (n_cols, len(list_proj)) where each column corresponds to a
            projection, and the entries are 0 or 1 based on the projections.
        """
        list_bi = [projection[1] for projection in list_proj]
        B = np.ones((len(list_proj), n_cols), dtype=int)

        for j in range(len(list_bi)):
            B[j, list_bi[j]] = 0

        return B.transpose()

    def _compute_partial_p_value(
            self,
            B: np.ndarray,
            U: np.ndarray,
            U_sigma: np.ndarray,
            k: int
        ) -> float:
        """
        Computes the partial p-value for a statistical test based on a given permutation.

        Parameters:
        -----------
        B : np.ndarray
            Pass matrix indicating the column used to create the target in each projection.
        U : np.ndarray
            A vector of shape (nb_permutations,) representing the observed test statistics.
        U_sigma : np.ndarray
            A matrix of shape (nb_permutations, nb_observations) where each row represents the test
            statistics for a given projection and all the permutations.
        k : int
            The index of the column on which to compute the partial p_value.

        Returns:
        --------
        float
            The partial p-value.
        """
        U_k = B[k, :]@U
        p_v_k = 1

        for u_sigma_k in (B[k, :]@U_sigma).tolist():
            if u_sigma_k >= U_k:
                p_v_k += 1

        return p_v_k / (self.nb_permutation + 1)

    def test(self, df: Union[pd.DataFrame, np.ndarray]) -> Union[float, Tuple[float, List[float]]]:
        """
        Apply the PKLM test over a real dataset.

        Parameters
        ----------
        df : np.ndarray
            The input dataset with missing values.

        Returns
        -------
        float
            If compute_partial_p_values=False. Returns the p-value of the test.
        Tuple[float, List[float]]
            If compute_partial_p_values=True. Returns the p-value of the test and the list of all
            the partial p-values.
        """
        df = self._pklm_preprocessing(df)
        self._check_nb_patterns(df)

        M = np.isnan(df).astype(int)
        list_proj = [self._draw_projection(df) for _ in range(self.nb_projections)]
        list_perm = [self.rng.permutation(M) for _ in range(self.nb_permutation)]
        U = 0.0
        list_U_sigma = [0.0 for _ in range(self.nb_permutation)]

        parallel_results = Parallel(n_jobs=-1)(
            delayed(self._parallel_process_projection)(
                df, list_perm, features_idx, target_idx
            )
            for features_idx, target_idx in list_proj
        )

        for U_projection, results in parallel_results:
            U += U_projection
            list_U_sigma = [x + y for x, y in zip(list_U_sigma, results)]

        # Je suggère d'alléger le code de cette manipulation même si théoriquement ça a de la valeur
        U = U / self.nb_projections
        list_U_sigma = [x / self.nb_permutation for x in list_U_sigma]

        p_value = 1
        for u_sigma in list_U_sigma:
            if u_sigma >= U:
                p_value += 1

        p_value = p_value / (self.nb_permutation + 1)
        
        if not self.compute_partial_p_values:
            return p_value
        else:
            _, n_cols = df.shape
            B = self._build_B(list_proj, n_cols)
            U = np.array([item[0] for item in parallel_results])
            U_sigma = np.array([item[1] for item in parallel_results])
            p_values = [self._compute_partial_p_value(B, U, U_sigma, k) for k in range(n_cols)]
            return p_value, p_values
