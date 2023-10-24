from __future__ import annotations

import functools
from typing import Callable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn import utils as sku
from sklearn.utils import resample
import math
from scipy import optimize
from scipy.special import expit

from qolmat.utils.exceptions import NoMissingValue, SubsetIsAString


def compute_transition_counts_matrix(states: pd.Series):
    if isinstance(states.iloc[0], tuple):
        n_variables = len(states.iloc[0])
        state_nonan = pd.Series([tuple([False] * n_variables)])
    else:
        n_variables = 1
        state_nonan = pd.Series([False])
    states = pd.concat([state_nonan, states, state_nonan], ignore_index=True)
    df_couples = pd.DataFrame({"current": states, "next": states.shift(-1)})
    df_couples = df_couples.iloc[:-1]
    counts = df_couples.groupby(["current", "next"]).size()
    df_counts = counts.unstack().fillna(0)
    return df_counts


def compute_transition_matrix(states: pd.Series, ngroups: Optional[List] = None):
    if ngroups is None:
        df_counts = compute_transition_counts_matrix(states)
    else:
        list_counts = [compute_transition_counts_matrix(df) for _, df in states.groupby(ngroups)]
        df_counts = functools.reduce(lambda a, b: a.add(b, fill_value=0), list_counts)

    df_transition = df_counts.div(df_counts.sum(axis=1), axis=0)
    return df_transition


def get_sizes_max(values_isna: pd.Series) -> pd.Series[int]:
    ids_hole = (values_isna.diff() != 0).cumsum()
    sizes_max = values_isna.groupby(ids_hole, group_keys=True).apply(
        lambda x: (~x) * np.arange(len(x))
    )
    sizes_max = sizes_max.shift(1)
    sizes_max = sizes_max.fillna(0)
    sizes_max = sizes_max.astype(int)
    return sizes_max


def pick_coeffs(
    X: np.ndarray,
    idxs_obs: np.ndarray,
    idxs_nas: np.ndarray,
    self_mask: bool = False,
) -> np.ndarray:
    n, d = X.shape
    if self_mask:
        coeffs = np.random.rand(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.rand(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(
    X: np.ndarray, coeffs: np.ndarray, p: float, self_mask: bool = False
) -> np.ndarray:
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):

            def f(x: np.ndarray) -> np.ndarray:
                return expit(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -1000, 1000)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):

            def f(x: np.ndarray) -> np.ndarray:
                return expit(np.dot(X, coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -1000, 1000)
    return intercepts


class _HoleGenerator:
    """
    This abstract class implements the generic method to generate masks according to law of missing
    values.

    Parameters
    ----------
    n_splits : int
        number of dataframes with missing additional missing values to be created
    subset : Optional[List[str]]
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float]
        Ratio of values ​​to mask, by default 0.05.
    random_state : Optional[int]
        The seed used by the random number generator, by default 42.
    groups: Tuple[str, ...]
        Column names used to group the data
    """

    generate_mask: Callable
    ngroups: Optional[List]

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
        groups: Tuple[str, ...] = (),
    ) -> None:
        self.n_splits = n_splits
        self.subset = subset
        self.ratio_masked = ratio_masked
        self.random_state = random_state
        self.groups = groups

    def fit(self, X: pd.DataFrame) -> _HoleGenerator:
        """
        Fits the generator.

        Parameters
        ----------
        X : pd.DataFrame
            Initial dataframe with a missing pattern to be imitated.
        """
        self._check_subset(X)
        self.dict_ratios = {}
        missing_per_col = X[self.subset].isna().sum()
        self.dict_ratios = (missing_per_col / missing_per_col.sum()).to_dict()
        if self.groups:
            self.ngroups = X.groupby(list(self.groups)).ngroup().rename("_ngroup")
        else:
            self.ngroups = None

        return self

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """Create a list of boolean masks representing the data to mask.
        Parameters
        ----------
        X : pd.DataFrame
            Initial dataframe with a missing pattern to be imitated.

        Returns
        -------
        Dict[str, pd.DataFrame]
            the initial dataframe, the dataframe with additional missing entries and the created
            mask
        """

        self.fit(X)
        list_masks = []
        for _ in range(self.n_splits):
            if self.ngroups is None:
                mask = self.generate_mask(X)
            else:
                mask = X.groupby(self.ngroups, group_keys=False).apply(self.generate_mask)
            list_masks.append(mask)
        return list_masks

    def _check_subset(self, X: pd.DataFrame):
        columns_with_nans = X.columns[X.isna().any()]
        if self.subset is None:
            self.subset = columns_with_nans
        elif isinstance(self.subset, str):
            raise SubsetIsAString(self.subset)
        else:
            subset_without_nans = [
                column for column in self.subset if column not in columns_with_nans
            ]
            if len(subset_without_nans) > 0:
                raise NoMissingValue(subset_without_nans)


class UniformHoleGenerator(_HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated randomly, using the resample method of scikit learn.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=(),
        )

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a mask for the dataframe at hand.

        Parameters
        ----------
        X : pd.DataFrame
            Initial dataframe with a missing pattern to be imitated.
        """

        self.rng = sku.check_random_state(self.random_state)
        df_mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        n_masked_col = math.ceil(self.ratio_masked * len(X))

        for column in self.subset:
            indices = np.where(X[column].notna())[0]
            indices = resample(
                indices,
                replace=False,
                n_samples=n_masked_col,
                stratify=None,
            )
            df_mask[column].iloc[indices] = True

        return df_mask


class _SamplerHoleGenerator(_HoleGenerator):
    """This abstract class implements a generic way to generate holes in a dataframe by sampling 1D
     hole size distributions.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    groups: Tuple[str, ...]
        Column names used to group the data
    """

    sample_sizes: Callable

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
        groups: Tuple[str, ...] = (),
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=groups,
        )

    def generate_hole_sizes(self, column: str, n_masked: int, sort: bool = True) -> List[int]:
        """Generate a sequence of states "states" of size "size" from
        a transition matrix "df_transition"

        Parameters
        ----------
        size : int
            length of the output sequence

        Returns
        -------
        List[float]
        """
        sizes_sampled = self.sample_sizes(column, n_masked)
        sizes_sampled = sizes_sampled[sizes_sampled.cumsum() < n_masked]
        n_masked_sampled = sizes_sampled.sum()
        list_sizes = sizes_sampled.tolist() + [n_masked - n_masked_sampled]
        if sort:
            list_sizes = sorted(list_sizes, reverse=True)
        return list_sizes

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create missing data in an arraylike object based on a markov chain.
        States of the MC are the different masks of missing values:
        there are at most pow(2,X.shape[1]) possible states.

        Parameters
        ----------
        X : pd.DataFrame
            initial dataframe with missing (true) entries

        Returns
        -------
        mask : pd.DataFrame
            masked dataframe with additional missing entries
        """
        mask = pd.DataFrame(False, columns=X.columns, index=X.index)
        n_masked_col = round(self.ratio_masked * len(X))
        list_failed: List = []
        self.rng = sku.check_random_state(self.random_state)
        for column in self.subset:
            states = X[column].isna()

            sizes_max = get_sizes_max(states)
            n_masked_left = n_masked_col

            sizes_sampled = self.generate_hole_sizes(column, n_masked_col, sort=True)
            assert sum(sizes_sampled) == n_masked_col
            sizes_sampled += self.generate_hole_sizes(column, n_masked_col, sort=False)
            for sample in sizes_sampled:
                sample = min(min(sample, sizes_max.max()), n_masked_left)
                i_hole = self.rng.choice(np.where(sample <= sizes_max)[0])

                assert (~mask[column].iloc[i_hole - sample : i_hole]).all()
                mask[column].iloc[i_hole - sample : i_hole] = True
                n_masked_left -= sample

                sizes_max.iloc[i_hole - sample : i_hole] = 0
                sizes_max.iloc[i_hole:] = np.minimum(
                    sizes_max.iloc[i_hole:],
                    np.arange(len(sizes_max.iloc[i_hole:])),
                )
                if n_masked_left == 0:
                    break

        if list_failed:
            warnings.warn(f"No place to introduce sampled holes of size {list_failed}!")
        return mask


class GeometricHoleGenerator(_SamplerHoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated following a Markov 1D process.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : Union[None, int, np.random.RandomState], optional
        The seed used by the random number generator, by default 42.
    groups: Tuple[str, ...]
        Column names used to group the data
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
        groups: Tuple[str, ...] = (),
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=groups,
        )

    def fit(self, X: pd.DataFrame) -> GeometricHoleGenerator:
        """
        Get the transition matrix from a list of states

        Parameters
        ----------
        X : pd.DataFrame
            transition matrix (stochastic matrix) current in index, next in columns
            1 is missing


        Returns
        -------
        Markov1DHoleGenerator
            The model itself

        """
        super().fit(X)
        self.dict_probas_out = {}
        for column in self.subset:
            states = X[column].isna()
            df_transition = compute_transition_matrix(states, self.ngroups)
            self.dict_probas_out[column] = df_transition.loc[True, False]

        return self

    def sample_sizes(self, column, n_masked):
        proba_out = self.dict_probas_out[column]
        mean_size = 1 / proba_out
        n_holes = 2 * round(n_masked / mean_size)
        sizes_sampled = pd.Series(self.rng.geometric(p=proba_out, size=n_holes))
        return sizes_sampled


class EmpiricalHoleGenerator(_SamplerHoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The distribution of holes is learned from the data.
    The distributions are learned column by column.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    groups: Tuple[str, ...]
        Column names used to group the data
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
        groups: Tuple[str, ...] = (),
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=groups,
        )

    def compute_distribution_holes(self, states: pd.Series) -> pd.Series:
        series_id = (states.diff() != 0).cumsum()
        series_id = series_id[states]
        distribution_holes = series_id.value_counts().value_counts()
        distribution_holes.index.name = "_size_hole"
        # distribution_holes /= distribution_holes.sum()
        return distribution_holes

    def fit(self, X: pd.DataFrame) -> EmpiricalHoleGenerator:
        """Compute the holes sizes of a dataframe.
        Dataframe df has only one column

        Parameters
        ----------
        X : pd.DataFrame
            data with holes

        Returns
        -------
        EmpiricalTimeHoleGenerator
            The model itself
        """
        super().fit(X)

        self.dict_distributions_holes = {}
        for column in self.subset:
            states = X[column].isna()
            if self.ngroups is None:
                self.dict_distributions_holes[column] = self.compute_distribution_holes(states)
            else:
                distributions_holes = states.groupby(self.ngroups).apply(
                    self.compute_distribution_holes
                )
                distributions_holes = distributions_holes.groupby(by="_size_hole").sum()
                self.dict_distributions_holes[column] = distributions_holes
        return self

    def sample_sizes(self, column, n_masked):
        """Create missing data in an arraylike object based on the holes size distribution.

        Parameters
        ----------
        column : str
            name of the column to fill with holes
        nb_holes : Optional[int], optional
            number of holes to create, by default 10

        Returns
        -------
        samples_sizes : List[int]
        """
        distribution_holes = self.dict_distributions_holes[column]
        distribution_holes /= distribution_holes.sum()
        mean_size = (distribution_holes.values * distribution_holes.index.values).sum()

        n_samples = 2 * round(n_masked / mean_size)
        sizes_sampled = self.rng.choice(distribution_holes.index, n_samples, p=distribution_holes)
        return sizes_sampled


class MultiMarkovHoleGenerator(_HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated according to a Markov process.
    Each line of the dataframe mask (np.nan) represents a state of the Markov chain.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values to add, by default 0.05
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    groups: Tuple[str, ...]
        Column names used to group the data
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
        groups: Tuple[str, ...] = (),
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=groups,
        )

    def fit(self, X: pd.DataFrame) -> MultiMarkovHoleGenerator:
        """
        Get the transition matrix from a list of states
        transition matrix (stochastic matrix) current in index,
        next in columns 1 is missing

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        MultiMarkovHoleGenerator
            The model itself

        """
        super().fit(X)

        states = X[self.subset].isna().apply(lambda x: tuple(x), axis=1)
        self.df_transition = compute_transition_matrix(states, self.ngroups)
        self.df_transition.index = pd.MultiIndex.from_tuples(self.df_transition.index)
        self.df_transition.columns = pd.MultiIndex.from_tuples(self.df_transition.columns)

        return self

    def generate_multi_realisation(self, n_masked: int) -> List[List[Tuple[bool, ...]]]:
        """Generate a sequence of states "states" of size "size"
        from a transition matrix "df_transition"

        Parameters
        ----------
        df_transition : pd.DataFrame
            transition matrix (stochastic matrix)
        size : int
            length of the output sequence

        Returns
        -------
        realisation ; List[int]
            sequence of states
        """
        states = sorted(list(self.df_transition.index))
        state_nona = tuple([False] * len(states[0]))

        state = state_nona
        realisations = []
        count_masked = 0
        while count_masked < n_masked:
            realisation = []
            while True:
                probas = self.df_transition.loc[state, :].values
                state = np.random.choice(self.df_transition.columns, 1, p=probas)[0]
                if state == state_nona:
                    break
                else:
                    count_masked += sum(state)
                    realisation.append(state)
            if realisation:
                realisations.append(realisation)
        return realisations

    def generate_mask(self, X: pd.DataFrame) -> List[pd.DataFrame]:
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
            the initial dataframe, the dataframe with additional missing entries and the created
            mask
        """

        self.rng = sku.check_random_state(self.random_state)
        X_subset = X[self.subset]
        mask = pd.DataFrame(False, columns=X_subset.columns, index=X_subset.index)

        values_hasna = X_subset.isna().any(axis=1)

        sizes_max = get_sizes_max(values_hasna)
        n_masked_left = int(X[self.subset].size * self.ratio_masked)

        realisations = self.generate_multi_realisation(n_masked_left)
        realisations = sorted(realisations, reverse=True)
        for realisation in realisations:
            size_hole = len(realisation)
            n_masked = sum([sum(row) for row in realisation])
            size_hole = min(size_hole, sizes_max.max())
            realisation = realisation[:size_hole]
            i_hole = self.rng.choice(np.where(size_hole <= sizes_max)[0])
            assert (~mask.iloc[i_hole - size_hole : i_hole]).all().all()
            if size_hole != 0:
                mask.iloc[i_hole - size_hole : i_hole] = mask.iloc[
                    i_hole - size_hole : i_hole
                ].where(~np.array(realisation).astype(bool), other=True)
            n_masked_left -= n_masked

            sizes_max.iloc[i_hole - size_hole : i_hole] = 0
            sizes_max.iloc[i_hole:] = np.minimum(
                sizes_max.iloc[i_hole:],
                np.arange(len(sizes_max.iloc[i_hole:])),
            )
            if n_masked_left <= 0:
                break

        complete_mask = pd.DataFrame(False, columns=X.columns, index=X.index)
        complete_mask[self.subset] = mask[self.subset]
        return mask


class GroupedHoleGenerator(_HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated from groups, specified by the user.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked to add, by default 0.05
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    groups : Tuple[str, ...]
        Names of the columns forming the groups, by default []
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
        groups: Tuple[str, ...] = (),
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            ratio_masked=ratio_masked,
            random_state=random_state,
            groups=groups,
        )

        if groups == ():
            raise Exception("Argument groups is an empty tuple!")

    def fit(self, X: pd.DataFrame) -> GroupedHoleGenerator:
        """Create the groups based on the column names (groups attribute)

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        GroupedHoleGenerator
            The model itself

        Raises
        ------
        if the number of samples/splits is greater than the number of groups.
        """

        super().fit(X)

        if self.n_splits > self.ngroups.nunique():
            raise ValueError("n_samples has to be smaller than the number of groups.")

        return self

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """creates masked dataframes

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        List[pd.DataFrame]
            list of masks
        """
        self.fit(X)
        group_sizes = X.groupby(self.ngroups, group_keys=False).count().mean(axis=1)
        list_masks = []

        for _ in range(self.n_splits):
            shuffled_group_sizes = group_sizes.sample(frac=1)

            ratio_masks = shuffled_group_sizes.cumsum() / len(X)
            ratio_masks = ratio_masks.reset_index(name="ratio")

            closest_ratio_mask = ratio_masks.iloc[
                (ratio_masks["ratio"] - self.ratio_masked).abs().argsort()[:1]
            ]
            groups_masked = ratio_masks.iloc[: closest_ratio_mask.index[0], :]["_ngroup"].values
            if closest_ratio_mask.index[0] == 0:
                groups_masked = ratio_masks.iloc[:1, :]["_ngroup"].values

            df_mask = pd.DataFrame(
                False,
                columns=X.columns,
                index=X.index,
            )

            ngroups = pd.Series(self.ngroups)
            observed_indices = ngroups[ngroups.isin(groups_masked)].index
            df_mask.loc[observed_indices, self.subset] = True

            df_mask[X.isna()] = False

            list_masks.append(df_mask)

        return list_masks


class MCAR(UniformHoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated randomly, using the resample method of scikit learn.

    Parameters
    ----------
    n_splits : int, optional
        Number of splits, by default 1.
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int = 1,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__(
            n_splits=n_splits, subset=subset, ratio_masked=ratio_masked, random_state=random_state
        )

    def _check_subset(self, X: pd.DataFrame):
        if self.subset is None:
            self.subset = X.columns
        if isinstance(self.subset, str):
            raise SubsetIsAString(self.subset)


class MAR(_HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    Missing at random mechanism with a logistic masking model.
    First, a subset of variables with *no* missing values is randomly selected.
    The remaining variables have missing values according to a logistic model
    with random weights, re-scaled so as to attain the desired proportion
    of missing values on those variables.
    This class is based on the function MAR_mask in
    https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/utils/simulate.py


    Parameters
    ----------
    n_splits : int, optional
        Number of splits, by default 1.
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    ratio_observed : Optional[float], optional
        Proportion of variables with *no* missing values that will be used
        for the logistic masking model, by default 0.1.
    sample_columns: bool, optional
        Sample variables that will all be observed, by default True.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int = 1,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        ratio_observed: float = 0.1,
        sample_columns: bool = True,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__(
            n_splits=n_splits, subset=subset, ratio_masked=ratio_masked, random_state=random_state
        )
        self.ratio_observed = ratio_observed
        self.sample_columns = sample_columns

    def _check_subset(self, X: pd.DataFrame):
        if self.subset is None:
            self.subset = X.columns
        if isinstance(self.subset, str):
            raise SubsetIsAString(self.subset)

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        df_mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        n, d = X.shape

        d_obs = max(
            int(self.ratio_observed * d), 1
        )  # number of variables that will have no missing values (at least one variable)
        d_na = d - d_obs  # number of variables that will have missing values

        # Sample variables that will all be observed, and those with missing values:
        if self.sample_columns:
            idxs_obs = np.random.choice(d, d_obs, replace=False)
        else:
            idxs_obs = np.array(list(range(d_obs)))

        idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

        # Other variables will have NA proportions that depend on those observed variables,
        # through a logistic model
        # The parameters of this logistic model are random.

        # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        coeffs = pick_coeffs(X.values, idxs_obs, idxs_nas)
        # Pick the intercepts to have a desired amount of missing values
        intercepts = fit_intercepts(X.iloc[:, idxs_obs].values, coeffs, self.ratio_masked)

        ps = expit(X.iloc[:, idxs_obs].values @ coeffs + intercepts)

        ber = np.random.rand(n, d_na)
        df_mask.iloc[:, idxs_nas] = ber < ps

        return df_mask


class MNAR(_HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    Missing not at random mechanism with a logistic masking model.
    It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs.
    Hence, values that are inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing
    probabilities are determined by the logistic model.
    Then inputs are then masked MCAR (hence, missing values from the second set will depend on
    masked values.
    In either case, weights are random and the intercept is selected to attain the desired
    proportion of missing values.
    This class is based on the function MNAR_mask_logistic in
    https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/utils/simulate.py


    Parameters
    ----------
    n_splits : int, optional
        Number of splits, by default 1.
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    ratio_variable :Optional[float], optional
        Proportion of variables that will be used for the logistic masking model
        (only if exclude_inputs), by default 0.3.
    exclude_inputs :
        True: mechanism (ii) is used, False: (i)
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int = 1,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        ratio_variable: float = 0.3,
        exclude_inputs: bool = True,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):
        super().__init__(
            n_splits=n_splits, subset=subset, ratio_masked=ratio_masked, random_state=random_state
        )
        self.ratio_variable = ratio_variable
        self.exclude_inputs = exclude_inputs

    def _check_subset(self, X: pd.DataFrame):
        if self.subset is None:
            self.subset = X.columns
        if isinstance(self.subset, str):
            raise SubsetIsAString(self.subset)

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        df_mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        n, d = X.shape
        d_params = (
            max(int(self.ratio_variable * d), 1) if self.exclude_inputs else d
        )  # number of variables used as inputs (at least 1)
        d_na = (
            d - d_params if self.exclude_inputs else d
        )  # number of variables masked with the logistic model

        # Sample variables that will be parameters for the logistic regression:
        idxs_params = (
            np.random.choice(d, d_params, replace=False) if self.exclude_inputs else np.arange(d)
        )
        idxs_nas = (
            np.array([i for i in range(d) if i not in idxs_params])
            if self.exclude_inputs
            else np.arange(d)
        )

        # Other variables will have NA proportions selected by a logistic model
        # The parameters of this logistic model are random.

        # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        coeffs = pick_coeffs(X.values, idxs_params, idxs_nas)
        # Pick the intercepts to have a desired amount of missing values
        intercepts = fit_intercepts(X.iloc[:, idxs_params].values, coeffs, self.ratio_masked)

        ps = expit(X.iloc[:, idxs_params].values @ coeffs + intercepts)

        ber = np.random.rand(n, d_na)
        df_mask.iloc[:, idxs_nas] = ber < ps

        # If the inputs of the logistic model are excluded from MNAR missingness,
        # mask some values used in the logistic model at random.
        # This makes the missingness of other variables potentially dependent on masked values

        if self.exclude_inputs:
            df_mask.iloc[:, idxs_params] = np.random.rand(n, d_params) < self.ratio_masked

        return df_mask
