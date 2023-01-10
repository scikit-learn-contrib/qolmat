from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def compute_transition_matrix(states: pd.Series):
    df_couples = pd.DataFrame({"current": states, "next": states.shift(-1)})
    counts = df_couples.groupby(["current", "next"]).size()
    df_counts = counts.unstack().fillna(0)
    df_transition = df_counts.div(df_counts.sum(axis=1), axis=0)
    return df_transition


class HoleGenerator:
    """
    This class implements a method to get indices of observed and missing values.

    Parameters
    ----------
    n_splits : int
        number of dataframes with missing additional missing values to be created
    subset : Optional[List[str]]
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float]
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int]
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
        groups: Optional[List[str]] = [],
    ) -> None:
        self.n_splits = n_splits
        self.subset = subset
        self.ratio_missing = ratio_missing
        self.random_state = random_state
        self.groups = groups

    def fit(self, X: pd.DataFrame) -> HoleGenerator:
        self._check_subset(X)
        self.dict_ratios = {}
        missing_per_col = X[self.subset].isna().sum()
        self.dict_ratios = (missing_per_col / missing_per_col.sum()).to_dict()
        if self.groups == []:
            self.ngroups = None
        else:
            self.ngroups = X.groupby(self.groups).ngroup()

        return self

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:
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

        self.fit(X)
        list_masks = []
        for _ in range(self.n_splits):
            if self.ngroups is None:
                mask = self.generate_mask(X)
            else:
                mask = X.groupby(self.ngroups).apply(self.generate_mask)
            list_masks.append(mask)
        return list_masks

    def _check_subset(self, X: pd.DataFrame):
        columns_with_nans = X.columns[X.isna().any()]
        if self.subset is None:
            self.subset = columns_with_nans
        else:
            subset_without_nans = [
                column for column in self.subset if column not in columns_with_nans
            ]
            if len(subset_without_nans) > 0:
                raise Exception(
                    f"No missing value in the columns {subset_without_nans}!"
                    "You need to pass the relevant column name in the subset argument!"
                )


class UniformHoleGenerator(HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated randomly, using the resample method of scikit learn.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float], optional
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
            groups=[],
        )

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:

        df_mask = pd.DataFrame(False, index=X.index, columns=X.columns)
        n_missing_col = round(self.ratio_missing * len(X))

        for column in self.subset:

            indices = np.where(X[column].notna())[0]
            indices = resample(
                indices,
                replace=False,
                n_samples=n_missing_col,
                stratify=None,
            )
            df_mask[column].iloc[indices] = True

        return df_mask


class ColWiseSamplerHoleGenerator(HoleGenerator):
    """This class implements a way to generate holes in a dataframe.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float], optional
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
        groups: Optional[List[str]] = [],
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
            groups=groups,
        )

    
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
        n_missing_col = round(self.ratio_missing * len(X))
        list_failed = []
        for column in self.subset:
            states = X[column].isna()

            ids_hole = (states.diff() != 0).cumsum()
            sizes_max = states.groupby(ids_hole).apply(lambda x: (~x) * np.arange(len(x))).shift(1).fillna(0).astype(int)
            n_masked_left = n_missing_col

            sizes_sampled = self.generate_hole_sizes(column, n_missing_col, sort=True)
            assert sum(sizes_sampled) == n_missing_col
            sizes_sampled += self.generate_hole_sizes(column, n_missing_col, sort=False)
            
            for sample in sizes_sampled:

                sample = min(min(sample, sizes_max.max()), n_masked_left)
                i_hole = np.random.choice(np.where(sample <= sizes_max)[0])

                assert (~mask[column].iloc[i_hole - sample : i_hole]).all()
                mask[column].iloc[i_hole - sample : i_hole] = True
                n_masked_left -= sample

                sizes_max.iloc[i_hole - sample : i_hole] = 0
                sizes_max.iloc[i_hole:] = np.minimum(sizes_max.iloc[i_hole:], np.arange(len(sizes_max.iloc[i_hole:])))
                if n_masked_left == 0:
                    break

        if list_failed:
            logger.warning(f"No place to introduce sampled holes of size {list_failed}!")
        return mask


class Markov1DHoleGenerator(ColWiseSamplerHoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated following a Markov 1D process.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float], optional
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
        groups: Optional[List[str]] = [],
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
            groups=groups,
        )

    def fit(self, X: pd.DataFrame) -> Markov1DHoleGenerator:
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
        # self._check_subset(X)
        self.dict_probas_out = {}
        for column in self.subset:
            states = X[column].isna()
            df_transition = compute_transition_matrix(states)
            self.dict_probas_out[column] = df_transition.loc[True, False]

        return self

    def generate_hole_sizes(self, column: str, n_missing: int, sort: bool=True) -> List[int]:
        """Generate a sequence of states "states" of size "size" from a transition matrix "df_transition"

        Parameters
        ----------
        size : int
            length of the output sequence

        Returns
        -------
        List[float]
        """
        proba_out = self.dict_probas_out[column]
        mean_size = 1 / proba_out
        n_holes = 2 * round(n_missing / mean_size)
        sizes_sampled = pd.Series(np.random.geometric(p=proba_out, size=n_holes))
        sizes_sampled = sizes_sampled[sizes_sampled.cumsum() < n_missing]
        n_missing_sampled = sizes_sampled.sum()
        list_sizes = sizes_sampled.tolist() + [n_missing - n_missing_sampled]
        if sort:
            list_sizes = sorted(list_sizes, reverse=True)
        return list_sizes


class EmpiricalTimeHoleGenerator(HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The distribution of holes is learned from the data.
    The distributions are learned column by column.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float], optional
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[str] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
        groups: Optional[List[str]] = [],
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
            groups=groups,
        )

    def fit(self, X: pd.DataFrame) -> EmpiricalTimeHoleGenerator:
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

        self._check_subset(X)

        self.dict_distributions_holes = {}
        for column in self.subset:
            df_count = X[[column]].copy()
            df_count["series_id"] = np.cumsum(df_count.isna().diff() != 0)
            df_count.loc[df_count[column].notna(), "series_id"] = 0
            df_count = df_count[df_count["series_id"] != 0]
            distribution_holes = df_count["series_id"].value_counts().value_counts()
            self.dict_distributions_holes[column] = distribution_holes

    def generate_hole_sizes(self, column: str, n_missing: Optional[int] = 10) -> List[int]:
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

        sizes_holes = self.dict_distributions_holes[column]
        samples_sizes = np.random.choice(
            sizes_holes.index, n_missing, p=sizes_holes / sum(sizes_holes)
        )
        return samples_sizes


class MultiMarkovHoleGenerator(HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated according to a Markov process.
    Each line of the dataframe mask (np.nan) represents a state of the Markov chain.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float], optional
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
        groups: Optional[List[str]] = [],
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
            groups=groups,
        )

    def fit(self, X: pd.DataFrame) -> MultiMarkovHoleGenerator:
        """
        Get the transition matrix from a list of states
        df_transition: pd.DataFrame
            transition matrix (stochastic matrix)
            current in index, next in columns
            1 is missing

        Parameters
        ----------
        states : pd.DataFrame

        Returns
        -------
        MultiMarkovHoleGenerator
            The model itself

        """
        self._check_subset(X)

        states = X[self.subset].isna().apply(lambda x: tuple(x), axis=1)
        self.df_transition = compute_transition_matrix(states)
        self.df_transition.index = pd.MultiIndex.from_tuples(self.df_transition.index)
        self.df_transition.columns = pd.MultiIndex.from_tuples(self.df_transition.columns)

        return self

    def generate_multi_realisation(self, n_missing: int) -> List[List[Tuple[bool]]]:
        """Generate a sequence of states "states" of size "size" from a transition matrix "df_transition"

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
        count_missing = 0
        while count_missing < n_missing:
            realisation = []
            while True:
                probas = self.df_transition.loc[state, :].values
                state = np.random.choice(self.df_transition.columns, 1, p=probas)[0]
                if state == state_nona:
                    break
                else:
                    count_missing += sum(state)
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

        X_subset = X[self.subset]
        mask = pd.DataFrame(False, columns=X_subset.columns, index=X_subset.index)

        mask_init = X_subset.isna().any(axis=1)
        n_missing = X[self.subset].size * self.ratio_missing

        realisations = self.generate_multi_realisation(n_missing)
        realisations = sorted(realisations, reverse=True)
        for realisation in realisations:
            size_hole = len(realisation)
            is_valid = (
                ~(mask_init | mask).T.all().rolling(size_hole + 2).max().fillna(1).astype(bool)
            )
            if not np.any(is_valid):
                logger.warning(f"No place to introduce sampled hole of size {size_hole}!")
                continue
            i_hole = np.random.choice(np.where(is_valid)[0])
            mask.iloc[i_hole - size_hole : i_hole] = mask.iloc[i_hole - size_hole : i_hole].where(
                ~np.array(realisation), other=True
            )

        complete_mask = pd.DataFrame(False, columns=X.columns, index=X.index)
        complete_mask[self.subset] = mask[self.subset]
        return mask


class GroupedHoleGenerator(HoleGenerator):
    """This class implements a way to generate holes in a dataframe.
    The holes are generated from groups, specified by the user.
    This class uses the GroupShuffleSplit function of sklearn.

    Parameters
    ----------
    n_splits : int
        Number of splits
    subset : Optional[List[str]], optional
        Names of the columns for which holes must be created, by default None
    ratio_missing : Optional[float], optional
        Ratio of missing values ​​to add, by default 0.05.
    random_state : Optional[int], optional
        The seed used by the random number generator, by default 42.
    groups : Optional[List[str]], optional
        Names of the columns forming the groups, by default empty list
    """

    def __init__(
        self,
        n_splits: int,
        groups: List[str],
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            ratio_missing=ratio_missing,
            random_state=random_state,
            groups=groups,
        )

        if groups == []:
            raise Exception("Argument groups is an empty list!")

    def fit(self, X: pd.DataFrame) -> GroupedHoleGenerator:
        """Creare the groups based on the column names (groups attribute)

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

        n_masked_per_column = len(X) * self.ratio_missing
        eligible_per_column = (X.notna().groupby(self.groups).all() * X.notna().groupby(self.groups).sum()).sum()
        self.probas_sample = n_masked_per_column / eligible_per_column

        if self.n_splits > self.ngroups.nunique():
            raise ValueError("n_samples has to be smaller than the number of groups.")

        return self

    def generate_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        mask = pd.DataFrame(False, columns=X.columns, index=X.index)

        for column in self.subset:
            has_no_nan = X[column].notna().all()
            is_sampled = np.random.uniform(0, 1) < self.probas_sample[column]

            if has_no_nan and is_sampled:
                mask[column] = True

        return mask
