"""Script for missing patterns."""

from __future__ import annotations

import functools
import math
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import utils as sku

from qolmat.utils.exceptions import SubsetIsAString
from qolmat.utils.utils import RandomSetting


def compute_transition_counts_matrix(states: pd.Series):
    """Compute transtion counts matrix.

    Parameters
    ----------
    states : pd.Series
        possible states (masks)

    Returns
    -------
    pd.Series | pd.DataFrame
        transition counts matrix

    """
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


def compute_transition_matrix(
    states: pd.Series, ngroups: Optional[List] = None
):
    """Compute the transition matrix.

    Parameters
    ----------
    states : pd.Series
        serie of possible states (masks)
    ngroups : Optional[List], optional
        groups, by default None

    Returns
    -------
    pd.DataFrame | pd.Series
        transition matrix

    """
    if ngroups is None:
        df_counts = compute_transition_counts_matrix(states)
    else:
        list_counts = [
            compute_transition_counts_matrix(df)
            for _, df in states.groupby(ngroups)
        ]
        df_counts = functools.reduce(
            lambda a, b: a.add(b, fill_value=0), list_counts
        )

    df_transition = df_counts.div(df_counts.sum(axis=1), axis=0)
    return df_transition


def get_sizes_max(values_isna: pd.Series) -> pd.Series[int]:
    """Get max sizes.

    Parameters
    ----------
    values_isna : pd.Series
        pandas series indicating if value is missing.

    """
    ids_hole = (values_isna.diff() != 0).cumsum()
    sizes_max = values_isna.groupby(ids_hole, group_keys=True).apply(
        lambda x: (~x) * np.arange(len(x))
    )
    sizes_max = sizes_max.shift(1)
    sizes_max = sizes_max.fillna(0)
    sizes_max = sizes_max.astype(int)
    return sizes_max


class _HoleGenerator:
    """Abstract HoleGenerator class.

    This abstract class implements the generic method to generate masks
    according to law of missing values.

    Parameters
    ----------
    n_splits : int
        number of dataframes with missing additional missing values to be
        created
    subset : Optional[List[str]]
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float]
        Ratio of values ​​to mask, by default 0.05.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
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
        random_state: RandomSetting = None,
        groups: Tuple[str, ...] = (),
    ) -> None:
        self.n_splits = n_splits
        self.subset = subset
        self.ratio_masked = ratio_masked
        self.random_state = random_state
        self.groups = groups

    def fit(self, X: pd.DataFrame) -> _HoleGenerator:
        """Fit the generator.

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
            self.ngroups = (
                X.groupby(list(self.groups)).ngroup().rename("_ngroup")
            )
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
            the initial dataframe, the dataframe with additional missing
            entries and the created mask

        """
        self.fit(X)
        list_masks = []
        for _ in range(self.n_splits):
            if self.ngroups is None:
                mask = self.generate_mask(X)
            else:
                mask = X.groupby(self.ngroups, group_keys=False).apply(
                    self.generate_mask
                )
            list_masks.append(mask)
        return list_masks

    def _check_subset(self, X: pd.DataFrame):
        columns_with_nans = X.columns[X.isna().any()]
        if self.subset is None:
            self.subset = columns_with_nans
        elif isinstance(self.subset, str):
            raise SubsetIsAString(self.subset)

    def save_rng_state(self):
        self.state_rng = self.random_state.get_state()

    def load_rng_state(self):
        self.random_state.set_state(self.state_rng)


class UniformHoleGenerator(_HoleGenerator):
    """UniformHoleGenerator class.

    This class implements a way to generate holes in a dataframe.
    The holes are generated randomly, using the resample method of sklearn.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    sample_proportional: bool, optional
        If True, generates holes in target columns with same equal frequency.
        If False, reproduces the empirical proportions between the variables.

    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: RandomSetting = None,
        sample_proportional: bool = False,
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=(),
        )
        self.sample_proportional = sample_proportional

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a mask for the dataframe at hand.

        Parameters
        ----------
        X : pd.DataFrame
            Initial dataframe with a missing pattern to be imitated.

        """
        self.random_state = sku.check_random_state(self.random_state)
        df_mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        for col in self.subset:
            ratio_masked = self.ratio_masked
            if self.sample_proportional:
                ratio_masked *= self.dict_ratios[col] * len(X.columns)
            n_masked_col = math.ceil(ratio_masked * len(X))
            indices_int = np.where(X[col].notna())[0]
            indices_int = self.random_state.choice(
                indices_int,
                replace=False,
                size=n_masked_col,
            )
            indices_int = df_mask.index[indices_int]
            df_mask.loc[indices_int, col] = True

        return df_mask


class _SamplerHoleGenerator(_HoleGenerator):
    """Abstract SamplerHoleGenerator class.

    This abstract class implements a generic way to generate holes in a
    dataframe by sampling 1D hole size distributions.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    groups: Tuple[str, ...]
        Column names used to group the data

    """

    sample_sizes: Callable

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: RandomSetting = None,
        groups: Tuple[str, ...] = (),
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_masked=ratio_masked,
            groups=groups,
        )

    def generate_hole_sizes(
        self, column: str, n_masked: int, sort: bool = True
    ) -> List[int]:
        """Generate a sequence of states "states" of size "size".

        Generated from a transition matrix "df_transition"

        Parameters
        ----------
        column : str
            column name
        n_masked: int
            number of masks
        sort: bool, optional
            true if sort, by default True

        Returns
        -------
        List[int]
            list of hole sizes

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

            sizes_sampled = self.generate_hole_sizes(
                column, n_masked_col, sort=True
            )
            if sum(sizes_sampled) != n_masked_col:
                raise ValueError(
                    "sum of sizes_sampled is different from n_masked_col: "
                    f"{sum(sizes_sampled)} != {n_masked_col}."
                )
            sizes_sampled += self.generate_hole_sizes(
                column, n_masked_col, sort=False
            )
            for sample in sizes_sampled:
                sample = min(min(sample, sizes_max.max()), n_masked_left)
                i_hole = self.rng.choice(np.where(sample <= sizes_max)[0])

                indices_hole = mask.index[i_hole - sample : i_hole]
                if not (~mask.loc[indices_hole, column]).all():
                    raise ValueError(
                        "The mask condition is not satisfied for "
                        f"column={column}, "
                        f"sample={sample}, "
                        f"and i_hole={i_hole}."
                    )

                mask.loc[indices_hole, column] = True
                n_masked_left -= sample

                sizes_max.iloc[i_hole - sample : i_hole] = 0
                sizes_max.iloc[i_hole:] = np.minimum(
                    sizes_max.iloc[i_hole:],
                    np.arange(len(sizes_max.iloc[i_hole:])),
                )
                if n_masked_left == 0:
                    break

        if list_failed:
            warnings.warn(
                f"No place to introduce sampled holes of size {list_failed}!"
            )
        return mask


class GeometricHoleGenerator(_SamplerHoleGenerator):
    """GeometricHoleGenerator class.

    This class implements a way to generate holes in a dataframe.
    The holes are generated following a Markov 1D process.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values ​​to add, by default 0.05.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    groups: Tuple[str, ...]
        Column names used to group the data

    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: RandomSetting = None,
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
        """Get the transition matrix from a list of states.

        Parameters
        ----------
        X : pd.DataFrame
            transition matrix (stochastic matrix) current in index,
            next in columns 1 is missing


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

    def sample_sizes(self, column: str, n_masked: int):
        """Sample sizes.

        Parameters
        ----------
        column : str
            column name
        n_masked : int
            number of masks

        Returns
        -------
        pd.Series
            sizes sampled

        """
        proba_out = self.dict_probas_out[column]
        mean_size = 1 / proba_out
        n_holes = 2 * round(n_masked / mean_size)
        sizes_sampled = pd.Series(
            self.rng.geometric(p=proba_out, size=n_holes)
        )
        return sizes_sampled


class EmpiricalHoleGenerator(_SamplerHoleGenerator):
    """EmpiricalHoleGenerator class.

    This class implements a way to generate holes in a dataframe.
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
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    groups: Tuple[str, ...]
        Column names used to group the data

    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: RandomSetting = None,
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
        """Compute the hole distribution.

        Parameters
        ----------
        states : pd.Series
            Series of states.

        Returns
        -------
        pd.Series
            hole distribution

        """
        series_id = (states.diff() != 0).cumsum()
        series_id = series_id[states]
        distribution_holes = series_id.value_counts().value_counts()
        distribution_holes.index.name = "_size_hole"
        # distribution_holes /= distribution_holes.sum()
        return distribution_holes

    def fit(self, X: pd.DataFrame) -> EmpiricalHoleGenerator:
        """Compute the holes sizes of a dataframe.

        Dataframe df has only one column.

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
                self.dict_distributions_holes[column] = (
                    self.compute_distribution_holes(states)
                )
            else:
                distributions_holes = states.groupby(self.ngroups).apply(
                    self.compute_distribution_holes
                )
                distributions_holes = distributions_holes.groupby(
                    by="_size_hole"
                ).sum()
                self.dict_distributions_holes[column] = distributions_holes
        return self

    def sample_sizes(self, column, n_masked):
        """Create missing data based on the holes size distribution.

        Parameters
        ----------
        column : str
            name of the column to fill with holes
        n_masked :int
            number of masks

        Returns
        -------
        samples_sizes : List[int]

        """
        distribution_holes = self.dict_distributions_holes[column]
        distribution_holes /= distribution_holes.sum()
        mean_size = (
            distribution_holes.values * distribution_holes.index.values
        ).sum()

        n_samples = 2 * round(n_masked / mean_size)
        sizes_sampled = self.rng.choice(
            distribution_holes.index, n_samples, p=distribution_holes
        )
        return sizes_sampled


class MultiMarkovHoleGenerator(_HoleGenerator):
    """MultiMarkovHoleGenerator class.

    This class implements a way to generate holes in a dataframe.
    The holes are generated according to a Markov process.
    Each line of the dataframe mask (np.nan) represents a state of the
    Markov chain.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked values to add, by default 0.05
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    groups: Tuple[str, ...]
        Column names used to group the data

    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: RandomSetting = None,
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
        """Get the transition matrix.

        Get the transition matrix from a list of states
        transition matrix (stochastic matrix) current in index,
        next in columns 1 is missing

        Parameters
        ----------
        X : pd.DataFrame
            input dataframe

        Returns
        -------
        MultiMarkovHoleGenerator
            The model itself

        """
        super().fit(X)

        states = X[self.subset].isna().apply(lambda x: tuple(x), axis=1)
        self.df_transition = compute_transition_matrix(states, self.ngroups)
        self.df_transition.index = pd.MultiIndex.from_tuples(
            self.df_transition.index
        )
        self.df_transition.columns = pd.MultiIndex.from_tuples(
            self.df_transition.columns
        )

        return self

    def generate_multi_realisation(
        self, n_masked: int
    ) -> List[List[Tuple[bool, ...]]]:
        """Generate a sequence of states "states" of size "size".

        Generated from a transition matrix "df_transition"

        Parameters
        ----------
        n_masked : int
            number of masks.

        Returns
        -------
        realisation ; List[int]
            sequence of states

        """
        states = sorted(self.df_transition.index)
        state_nona = tuple([False] * len(states[0]))

        state = state_nona
        realisations = []
        count_masked = 0
        while count_masked < n_masked:
            realisation = []
            while True:
                probas = self.df_transition.loc[state, :].values
                state = np.random.choice(
                    self.df_transition.columns, 1, p=probas
                )[0]
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
            the initial dataframe, the dataframe with additional missing
            entries and the created mask

        """
        self.rng = sku.check_random_state(self.random_state)
        X_subset = X[self.subset]
        mask = pd.DataFrame(
            False, columns=X_subset.columns, index=X_subset.index
        )

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
            if not (~mask.iloc[i_hole - size_hole : i_hole]).all().all():
                raise ValueError(
                    f"The mask condition is not satisfied for i_hole={i_hole} "
                    f"and size_hole={size_hole}."
                )
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
    """GroupedHoleGenerator class.

    This class implements a way to generate holes in a dataframe.
    The holes are generated from groups, specified by the user.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_masked : Optional[float], optional
        Ratio of masked to add, by default 0.05
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
        Pass an int for reproducible output across multiple function calls.
    groups : Tuple[str, ...]
        Names of the columns forming the groups, by default []

    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_masked: float = 0.05,
        random_state: RandomSetting = None,
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
        """Create the groups based on the column names (groups attribute).

        Parameters
        ----------
        X : pd.DataFrame
            input dataframe

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
            raise ValueError(
                "n_samples has to be smaller than the number of groups."
            )

        return self

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """Create masked dataframes.

        Parameters
        ----------
        X : pd.DataFrame
            input dataframe

        Returns
        -------
        List[pd.DataFrame]
            list of masks

        """
        self.fit(X)
        group_sizes = (
            X.groupby(self.ngroups, group_keys=False).count().mean(axis=1)
        )
        list_masks = []

        for _ in range(self.n_splits):
            shuffled_group_sizes = group_sizes.sample(
                frac=1, random_state=self.random_state
            )

            ratio_masks = shuffled_group_sizes.cumsum() / len(X)
            ratio_masks = ratio_masks.reset_index(name="ratio")

            closest_ratio_mask = ratio_masks.iloc[
                (ratio_masks["ratio"] - self.ratio_masked).abs().argsort()[:1]
            ]
            groups_masked = ratio_masks.iloc[: closest_ratio_mask.index[0], :][
                "_ngroup"
            ].values
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
