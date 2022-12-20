from typing import Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import GroupShuffleSplit


###########################################################################
# Missing data mechanisms depending on the categories of missing patterns #
###########################################################################


### missing at random


def MAR_mask(X: NDArray, p: float, p_obs: float) -> NDArray:
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : np.ndarray
        Data for which missing values will be simulated, shape (n, d)

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    mask = np.zeros(X.shape).astype(bool)

    d_obs = max(
        int(p_obs * d), 1
    )  ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  ## number of variables that wil have missing values

    ## sample variables that will all be observed, and those with missign values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ## other variables will have NA proportions depending on those observed ones, through a logistic model
    ## the parameters og this logistic model are random

    ## pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ## pick the intercepts to have a disered amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = 1 / (1 + np.exp(X[:, idxs_obs] @ coeffs + intercepts))

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


### missing not at random


def MNAR_mask_logistic(
    X: NDArray,
    p: float,
    p_params: Optional[float] = 0.3,
    exclude_inputs: Optional[bool] = True,
) -> NDArray:
    """Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.


    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float, optional
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs), by default 0.3
    exclude_inputs : bool, optional
        True: mechanism (ii) is used, False: (i), by default True

    Returns
    -------
    np.ndarray
        mask: Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    mask = np.zeros(X.shape).astype(bool)

    d_params = (
        max(int(p_params * d), 1) if exclude_inputs else d
    )  ## number of variables used as inputs (at least 1)
    d_na = (
        d - d_params if exclude_inputs else d
    )  ## number of variables masked with the logistic model

    ## sample variables that will be parameters for the LR
    idxs_params = (
        np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    )
    idxs_nas = (
        np.array([i for i in range(d) if i not in idxs_params])
        if exclude_inputs
        else np.arange(d)
    )

    ## other variables will have NA proportions selected bu a logistic model
    ## the params of this logistic model are random

    ## pick coefficietns so that W^Tx has unit variance (avoid shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ## pick the intercepts to have a disered amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = 1 / (1 + np.exp(X[:, idxs_params] @ coeffs + intercepts))

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = np.random.rand(n, d_params) < p

    return mask


def MNAR_self_mask_logistic(X: NDArray, p: float) -> NDArray:
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = 1 / (1 + np.exp(X * coeffs + intercepts))

    ber = np.random.rand(n, d)
    mask = ber < ps

    return mask


def MNAR_mask_quantiles(
    X: NDArray,
    p: float,
    q: float,
    p_params: float,
    cut: Optional[str] = "both",
    MCAR: Optional[bool] = False,
) -> NDArray:
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.

    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.

    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1)  ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(
        d, d_na, replace=False
    )  ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == "upper":
        quants = np.quantile(X[:, idxs_na], 1 - q, axis=0)
        m = X[:, idxs_na] >= quants
    elif cut == "lower":
        quants = np.quantile(X[:, idxs_na], q, axis=0)
        m = X[:, idxs_na] <= quants
    elif cut == "both":
        u_quants = np.quantile(X[:, idxs_na], 1 - q, axis=0)
        l_quants = np.quantile(X[:, idxs_na], q, axis=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = np.random.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
        ## Add a mcar mecanism on top
        mask = mask | (np.random.rand(n, d) < p)

    return mask


def pick_coeffs(
    X: NDArray,
    idxs_obs: Optional[NDArray] = None,
    idxs_nas: Optional[NDArray] = None,
    self_mask: Optional[bool] = False,
) -> NDArray:
    """Pick coefficients so that W^Tx has unit variance (avoids shrinking)

    Parameters
    ----------
    X : NDArray
        _description_
    idxs_obs : Optional[NDArray], optional
        _description_, by default None
    idxs_nas : Optional[NDArray], optional
        _description_, by default None
    self_mask : Optional[bool], optional
        _description_, by default False

    Returns
    -------
    NDArray
        _description_
    """
    n, d = X.shape
    if self_mask:
        coeffs = np.random.randn(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, axis=0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.randn(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(
    X: NDArray, coeffs: NDArray, p: float, self_mask: Optional[bool] = False
) -> NDArray:
    """pick the intercepts to have a disered amount of missing values

    Parameters
    ----------
    X : NDArray
        _description_
    coeffs : NDArray
        _description_
    p : float
        _description_
    self_mask : Optional[bool], optional
        _description_, by default False

    Returns
    -------
    NDArray
        _description_
    """
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):

            def f(x):
                return (1 / (1 + np.exp(X * coeffs[j] + x))).mean() - p

            intercepts[j] = scipy.optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):

            def f(x):
                return (1 / (1 + np.exp(X @ coeffs[:, j] + x))).mean() - p

            intercepts[j] = scipy.optimize.bisect(f, -50, 50)
    return intercepts


# Function produce_NA for generating missing values ------------------------------------------------------


def produce_NA_mechanism(
    X: NDArray,
    p_miss: float,
    mecha: Optional[str] = "MCAR",
    opt: Optional[str] = None,
    p_obs: Optional[float] = None,
    q: Optional[float] = None,
) -> Dict:
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str,
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

    Returns
    -------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(data=np.random.rand(20,4), columns=[f"var{i}" for i n range(1,5)])
    >>> # MCAR patterns
    >>> X_miss_mcar = missing_patterns.produce_NA(df, p_miss=0.4, mecha="MCAR")
    >>> X_mcar = X_miss_mcar['X_incomp']
    >>> R_mcar = X_miss_mcar['mask']
    >>> # MAR patterns
    >>> X_miss_mar = missing_patterns.produce_NA(df, p_miss=0.4, mecha="MAR", p_obs=0.5)
    >>> X_mar = X_miss_mar['X_incomp']
    >>> R_mar = X_miss_mar['mask']
    >>> # MNAR patterns with logistic model
    >>> X_miss_mnar = missing_patterns.produce_NA(df, p_miss=0.50, mecha="MNAR", opt="logistic", p_obs=0.2)
    >>> X_mar = X_miss_mnar['X_incomp']
    >>> R_mar = X_miss_mnar['mask']
    >>> # MNAR patterns with self masked model
    >>> X_miss_mnar = missing_patterns.produce_NA(df, p_miss=0.50, mecha="MNAR", opt="selfmasked")
    >>> X_mar = X_miss_mnar['X_incomp']
    >>> R_mar = X_miss_mnar['mask']
    >>> # MNAR patterns with quantiles
    >>> X_miss_mnar = missing_patterns.produce_NA(df, p_miss=0.50, mecha="MNAR", opt="quantile", p_obs=0.5, q=0.3)
    >>> X_mar = X_miss_mnar['X_incomp']
    >>> R_mar = X_miss_mnar['mask']
    """

    X_copy = X.copy()
    mask_init = np.isnan(X_copy)

    X_copy = X_copy.fillna(X_copy.median())
    if isinstance(X_copy, pd.DataFrame):
        X_nas = X_copy.values
    elif isinstance(X_copy, np.ndarray):
        X_nas = X_copy.copy()

    if mecha == "MAR":
        mask = MAR_mask(X_nas, p_miss, p_obs)
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X_nas, p_miss, p_obs)
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X_nas, p_miss, q, 1 - p_obs)
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X_nas, p_miss)
    else:
        mask = np.random.rand(X_nas.shape[0], X_nas.shape[1]) < p_miss

    X_nas[mask] = np.nan
    X_nas[mask_init] = np.nan
    mask[mask_init] = False  # initial missing values -> need to evaluate the imputation

    if isinstance(X, pd.DataFrame):
        X_nas = pd.DataFrame(X_nas, columns=X.columns, index=X.index)
        mask = pd.DataFrame(mask, columns=X.columns, index=X.index)

    return {"X_init": X, "X_incomp": X_nas, "mask": mask}


######################################################
# Missing data depending on the size of missing data #
######################################################


def transition_matrix(states: List[int]):
    """Get the transition matrix from a list of states

    Parameters
    ----------
    states : List[int]

    Returns
    -------
    T : np.ndarray
        transition matrix associatd to the states
    """
    n = 1 + max(states)
    T = np.zeros((n, n))
    for (i, j) in zip(states, states[1:]):
        T[i][j] += 1
    row_sums = T.sum(axis=1)
    T = T / row_sums[:, np.newaxis]
    return T


def generate_realisation(matrix: np.ndarray, states: List[int], length: int):
    """Generate a sequence of states "states" of length "length" from a transition matrix "matrix"

    Parameters
    ----------
    matrix : np.ndarray
        transition matrix (stochastic matrix)
    states : List[int]
        list of possible states
    length : int
        length of the output sequence

    Returns
    -------
    realisation ; List[int]
        sequence of states
    """
    states = sorted(list(set(states)))
    realisation = [np.random.choice(states)]
    for _ in range(length - 1):
        realisation.append(np.random.choice(states, 1, p=matrix[realisation[-1], :])[0])
    return realisation


def produce_NA_markov_chain(
    df: pd.DataFrame, columnwise_missing: Optional[bool] = False
) -> pd.DataFrame:
    """Create missing values based on markov chains

    Parameters
    ----------
    df : pd.DataFrame
        initial dataframe with missing values
    columnwise_missing : Optional[bool], optional
        True if each column has to be treated independently, by default False

    Returns
    -------
    mask : pd.DataFrame
        mask of missing values, True if missing, False if observed

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(data=np.random.rand(20,4), columns=[f"var{i}" for i n range(1,5)])
    >>> df = df.mask(np.random.random(df.shape) < 0.3) # already missing values since the idea is to artificially reproduce the same distribution of missing data sizes
    >>> res = produce_NA_markov_chain(df, columnwise_missing=False)


    """
    mask_init = np.isnan(df)
    mask = df.copy()

    if columnwise_missing:
        for column in df.columns:
            states = np.isnan(df[column]).astype(int).values
            T = transition_matrix(states)
            sample = generate_realisation(T, states, mask.shape[0])
            mask[column] = [bool((i + 1) % 2) for i in sample]
    else:
        u, states = np.unique(np.isnan(df), axis=0, return_inverse=True)
        T = transition_matrix(states)
        sample = generate_realisation(T, states, mask.shape[0])
        mask = u[sample]

    mask[mask_init] = False
    mask = pd.DataFrame(data=mask, columns=df.columns, index=df.index)

    return {"X_init": df, "X_incomp": df[~mask], "mask": mask}


# def produce_NA_():
#     C = np.cumsum(isnan.diff() != 0)
#     C[~isnan] = -1
#     C.value_.count().values().value_count()


# ------------------------------------------------------------------------------------------


class HoleGenerator:
    """
    This class implements a method to get indices of observed and missing values.
    """

    def __init__(
        self,
        method: str,
        n_splits: int,
        columnwise: Optional[bool] = True,
        column_groups: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
    ) -> None:
        self.method = method
        self.n_splits = n_splits
        self.columnwise = columnwise
        self.column_groups = column_groups
        self.ratio_missing = ratio_missing
        self.random_state = random_state

    @staticmethod
    def transition_matrix(states: List[int]) -> np.ndarray:
        """Get the transition matrix from a list of states

        Parameters
        ----------
        states : List[int]

        Returns
        -------
        T : np.ndarray
            transition matrix associatd to the states
        """
        n = 1 + max(states)
        T = np.zeros((n, n))
        for (i, j) in zip(states, states[1:]):
            T[i][j] += 1
        row_sums = T.sum(axis=1)
        T = T / row_sums[:, np.newaxis]
        return T

    @staticmethod
    def generate_realisation(
        matrix: np.ndarray, states: List[int], length: int
    ) -> List[int]:
        """Generate a sequence of states "states" of length "length" from a transition matrix "matrix"

        Parameters
        ----------
        matrix : np.ndarray
            transition matrix (stochastic matrix)
        states : List[int]
            list of possible states
        length : int
            length of the output sequence

        Returns
        -------
        realisation ; List[int]
            sequence of states
        """
        states = sorted(list(set(states)))
        realisation = [np.random.choice(states)]
        for _ in range(length - 1):
            realisation.append(
                np.random.choice(states, 1, p=matrix[realisation[-1], :])[0]
            )
        return realisation

    def _create_missing_markov(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create missing data in an arraylike object based on a markov chain.
        States of the MC are the different masks of missing values:
        there are at most pow(2,X.shape[1]) possible states.

        Parameters
        ----------
        X : pd.DataFrame
            initial dataframe with missing (true) entries

        Returns
        -------
        Dict[str, pd.DataFrame]
            the initial dataframe, the dataframe with additional missing entries and the created mask
        """
        mask_init = np.isnan(X)
        mask = X.copy()

        if self.columnwise:
            for column in X.columns:
                states = np.isnan(X[column]).astype(int).values
                T = transition_matrix(states)
                sample = generate_realisation(T, states, mask.shape[0])
                mask[column] = [bool((i + 1) % 2) for i in sample]
        else:
            u, states = np.unique(np.isnan(X), axis=0, return_inverse=True)
            T = transition_matrix(states)
            sample = generate_realisation(T, states, mask.shape[0])
            mask = u[sample]

        mask[mask_init] = False
        mask = pd.DataFrame(data=mask, columns=X.columns, index=X.index)

        return {"X_init": X, "X_incomp": X[~mask], "mask": mask}

    def _get_size_holes(self, df: pd.DataFrame) -> pd.Series:
        """Compute the holes sizes of a dataframe.
        Dataframe df has only one column

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with one column

        Returns
        -------
        sizes_holes : pd.Series
            index: hole size ; value: number of occurrences
        """
        df_ = df.copy()
        column_name = df_.columns[0]
        df_["series_id"] = np.cumsum(df_.isna().diff() != 0)
        df_.loc[df_[column_name].notna(), "series_id"] = 0
        df_ = df_.drop((df_[df_["series_id"] == 0]).index)
        sizes_holes = df_["series_id"].value_counts().value_counts()
        # sizes_holes = sizes_holes.reindex(np.arange(sizes_holes.max() + 1)).fillna(0)
        return sizes_holes

    def _create_missing_empirical(
        self, X: pd.DataFrame, nb_holes: Optional[int] = 10
    ) -> Dict[str, pd.DataFrame]:
        """Create missing data in an arraylike object based on the holes size distribution.

        Parameters
        ----------
        X : pd.DataFrame
            initial dataframe with missing (true) entries
        nb_holes : Optional[int], optional
            number of holes to create, by default 10

        Returns
        -------
        Dict[str, pd.DataFrame]
            the initial dataframe, the dataframe with additional missing entries and the created mask
        """

        mask_init = np.isnan(X)

        X_with_nan = X.copy()

        for column in X.columns:
            nb_missing = X[column].isna().sum()
            if nb_missing == 0:
                continue

            X_ = X[[column]]
            sizes_holes = self._get_size_holes(X_)

            if not nb_holes:
                pass  # TO DO

            chosen_sizes = np.random.choice(
                sizes_holes.index, nb_holes, p=sizes_holes / sum(sizes_holes)
            )
            s = 0
            for ind, chosen in enumerate(chosen_sizes):
                s += chosen
                if s > nb_missing:
                    break
            chosen_sizes = chosen_sizes[:ind]

            states = np.isnan(X[column]).astype(int).values
            T = transition_matrix(states)

            initial_indexes = X_.index.names
            X_ = X_.reset_index()

            for hole_size in chosen_sizes:
                for i in range(len(X_) - hole_size):
                    if (~np.isnan(X_.loc[i, column])) and (
                        np.random.uniform(0, 1) < T[1, 0]
                    ):
                        if X_.loc[i : i + hole_size, column].isna().sum() == 0:
                            X_.loc[i : i + hole_size, column] = np.nan
                            break
                    elif (np.isnan(X_.loc[i, column])) and (
                        np.random.uniform(0, 1) < T[0, 1]
                    ):
                        if X_.loc[i : i + hole_size, column].isna().sum() == 0:
                            X_.loc[i : i + hole_size, column] = np.nan
                            break

            X_with_nan[column] = X_[column].values

        mask = np.isnan(X_with_nan)
        mask[mask_init] = False

        return {"X_init": X, "X_incomp": X_with_nan, "mask": mask}

    def _create_groups(self, X: pd.DataFrame) -> None:
        """Creare the groups based on the column names (column_groups attribute)

        Parameters
        ----------
        X : pd.DataFrame

        Raises
        ------
        if the number of samples/splits is greater than the number of groups.
        """

        groups = X.groupby(self.column_groups).ngroup().values

        if self.n_splits > len(np.unique(groups)):
            raise ValueError("n_samples has to be smaller than the number of groups.")

        return groups

    def split(self, X: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Return a list of lists of training and testing indices.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        List[Tuple[pd.DataFrame, pd.DataFrame]] # # List[Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]]:
            List of length n_splits.
        """

        train_indices, test_indices = [], []
        mask_dfs, corrupted_dfs = [], []

        if self.method == "grouped":

            if self.column_groups is None:
                raise Exception("column_group is empty.")

            groups = self._create_groups(X)
            gss = GroupShuffleSplit(
                n_splits=self.n_splits,
                train_size=1 - self.ratio_missing,
                random_state=self.random_state,
            )
            for _, (observed_indices, missing_indices) in enumerate(
                gss.split(X=X, y=None, groups=groups)
            ):

                # create the boolean mask of missing values
                df_mask = pd.DataFrame(
                    data=np.full((X.shape), True),
                    columns=X.columns,
                    index=X.index,
                )
                df_mask.iloc[observed_indices, :] = False
                mask_dfs.append(df_mask)

                # create the corrupted (with artificial missing values) dataframe
                df_corrupted = X.copy()
                df_corrupted.iloc[missing_indices, :] = np.nan
                corrupted_dfs.append(df_corrupted)

                # train_indices.append(
                #     [(i, j) for i in observed_indices for j in range(X.shape[1])]
                # )
                # test_indices.append(
                #     [(i, j) for i in missing_indices for j in range(X.shape[1])]
                # )

        else:
            for _ in range(self.n_splits):
                if self.method == "markov":
                    results = self._create_missing_markov(X)
                elif self.method == "empirical":
                    results = self._create_missing_empirical(X)

                mask_dfs.append(results["mask"])
                corrupted_dfs.append(results["X_incomp"])

                # mask = results["mask"]
                # train_indices.append(
                #     [(i, j) for i, j in zip(np.where(~mask)[0], np.where(~mask)[1])]
                # )
                # test_indices.append(
                #     [(i, j) for i, j in zip(np.where(mask)[0], np.where(mask)[1])]
                # )

        return [
            (i, j) for i, j in zip(mask_dfs, corrupted_dfs)
        ]  # [(i, j) for i, j in zip(train_indices, test_indices)]
