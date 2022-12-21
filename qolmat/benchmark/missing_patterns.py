import logging

from typing import List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample

logger = logging.getLogger(__name__)


class HoleGenerator:
    """
    This class implements a method to get indices of observed and missing values.

    Parameters
    ----------
    n_splits : int
        number of dataframes with missing additional missing values to be created
    subset : Optional[List[str]]
        Names of the columns for which holes must be created
        By default None.
    ratio_missing : Optional[float]
        Ratio of missing values ​​to add
        By default 0.05.
    random_state : Optional[int]
        The seed used by the random number generator.
        By default 42.
    """

    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_splits = n_splits
        self.subset = subset
        self.ratio_missing = ratio_missing
        self.random_state = random_state


class RandomHoleGenerator(HoleGenerator):
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
        )

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:

        X_ = X[self.subset]

        mask_init = X_.isna()
        mask_init = mask_init.to_numpy().flatten()

        df_masks = []
        for _ in range(self.n_splits):
            indices = np.argwhere(mask_init > 0)[:, 0]
            indices = resample(
                indices,
                replace=False,
                n_samples=math.floor(len(indices) * self.ratio_missing),
                stratify=None,
            )

            mask = np.full(X_.shape, False)
            mask.flat[indices] = True
            mask = pd.DataFrame(
                mask.reshape(X_.shape), index=X_.index, columns=X_.columns
            )
            complete_mask = pd.DataFrame(False, columns=X.columns, index=X.index)
            complete_mask[self.subset] = mask[self.subset].values

            df_masks.append(complete_mask)

        return df_masks


class MarkovHoleGenerator(HoleGenerator):
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
        )

    @staticmethod
    def transition_matrix(states: pd.Series) -> pd.DataFrame:
        """Get the transition matrix from a list of states

        Parameters
        ----------
        states : pd.Series

        Returns
        -------
        df_transition : pd.DataFrame
            transition matrix associatd to the states
        """
        df_couples = pd.DataFrame({"current": states, "next": states.shift(1)})
        counts = df_couples.groupby(["current", "next"]).size()
        df_transition = counts.unstack().fillna(0)
        return df_transition.div(df_transition.sum(axis=1), axis=0)

    @staticmethod
    def generate_hole_sizes(df_transition: pd.DataFrame, size: int) -> List[int]:
        """Generate a sequence of states "states" of size "size" from a transition matrix "df_transition"

        Parameters
        ----------
        df_transition: pd.DataFrame
            transition matrix (stochastic matrix)
            current in index, next in columns
            1 is missing
        size : int
            length of the output sequence

        Returns
        -------
        List[float]
        """
        prob_out = df_transition.loc[False, True]
        return (np.random.poisson(lam=-np.log(prob_out), size=size) + 1).tolist()

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
            the initial dataframe, the dataframe with additional missing entries and the created mask
        """

        if self.subset is None:
            self.subset = X.columns

        list_masks = []
        for _ in range(self.n_splits):
            mask = pd.DataFrame(False, columns=X.columns, index=X.index)

            for column in self.subset:

                if X[column].isna().sum() == 0:
                    raise Exception(
                        "There is no missing value in the column. You need to pass the relevant column name in the subset argument!"
                    )

                states = X[column].isna()
                n_missing = round(len(X) * self.ratio_missing)
                df_transition = self.transition_matrix(states)
                samples_sizes = self.generate_hole_sizes(df_transition, n_missing)
                for sample in samples_sizes:
                    if sample > 0:
                        is_valid = (
                            ~(states | mask[column])
                            .rolling(sample + 2)
                            .max()
                            .fillna(1)
                            .astype(bool)
                        )
                        if not np.any(is_valid):
                            logger.warning("No place to introduce sampled hole!")
                            continue
                        i_hole = np.random.choice(np.where(is_valid)[0])
                        if mask[column].iloc[i_hole - sample + 1 : i_hole + 1].any():
                            print("overriding existing hole!")
                        mask[column].iloc[i_hole - sample : i_hole] = True
            list_masks.append(mask)
        return list_masks


class MultiMarkovHoleGenerator(MarkovHoleGenerator):
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
        )

    @staticmethod
    def generate_multi_realisation(
        df_transition: pd.DataFrame, n_missing: int
    ) -> List[List[Tuple[bool]]]:
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
        states = sorted(list(df_transition.index))
        state_nona = tuple([False] * len(states[0]))

        state = state_nona
        realisations = []
        count_missing = 0
        while count_missing < n_missing:
            realisation = []
            while True:
                state = np.random.choice(
                    df_transition.columns, 1, p=df_transition.loc[state, :].values
                )[0]
                if state == state_nona:
                    break
                else:
                    count_missing += sum(state)
                    realisation.append(state)
            if realisation:
                realisations.append(realisation)
        return realisations

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
            the initial dataframe, the dataframe with additional missing entries and the created mask
        """

        if self.subset is None:
            self.subset = X.columns

        list_masks = []
        n_missing = round(self.ratio_missing * X.size)
        for _ in range(self.n_splits):
            X_subset = X[self.subset]
            mask = pd.DataFrame(False, columns=X_subset.columns, index=X_subset.index)
            states = X_subset.isna().apply(lambda x: tuple(x), axis=1)
            df_states = pd.DataFrame(
                [[*a] for a in states.values],
                columns=X_subset.columns,
                index=X_subset.index,
            )

            df_transition = self.transition_matrix(states)
            df_transition.index = pd.MultiIndex.from_tuples(df_transition.index)
            realisations = self.generate_multi_realisation(df_transition, n_missing)
            for realisation in realisations:
                size_hole = len(realisation)
                is_valid = (
                    ~(df_states | mask)
                    .T.all()
                    .rolling(size_hole + 2)
                    .max()
                    .fillna(1)
                    .astype(bool)
                )
                if not np.any(is_valid):
                    logger.warning("No place to introduce sampled hole!")
                    continue
                i_hole = np.random.choice(np.where(is_valid)[0])
                mask.iloc[i_hole - size_hole : i_hole] = mask.iloc[
                    i_hole - size_hole : i_hole
                ].where(~np.array(realisation), other=True)

            complete_mask = pd.DataFrame(False, columns=X.columns, index=X.index)
            complete_mask[self.subset] = mask[self.subset]
            list_masks.append(complete_mask)
        return list_masks


class EmpiricalTimeHoleGenerator(HoleGenerator):
    def __init__(
        self,
        n_splits: int,
        subset: Optional[str] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
        )

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
        return sizes_holes

    def generate_hole_sizes(
        self, X: pd.DataFrame, nb_holes: Optional[int] = 10
    ) -> List[int]:
        """Create missing data in an arraylike object based on the holes size distribution.

        Parameters
        ----------
        X : pd.DataFrame
            initial dataframe with missing (true) entries
        nb_holes : Optional[int], optional
            number of holes to create, by default 10

        Returns
        -------
        samples_sizes : List[int]
        """

        nb_missing = X.isna().sum().sum()
        if nb_missing == 0:
            raise Exception("No missing value in the column!")

        sizes_holes = self._get_size_holes(X)
        samples_sizes = np.random.choice(
            sizes_holes.index, nb_holes, p=sizes_holes / sum(sizes_holes)
        )
        return samples_sizes

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:

        if self.subset is None:
            self.subset = X.columns

        list_masks = []
        for _ in range(self.n_splits):
            mask = pd.DataFrame(False, columns=X.columns, index=X.index)

            for column in self.subset:

                if X[column].isna().sum() == 0:
                    raise Exception(
                        "There is no missing value in the column. You need to pass the relevant column name in the subset argument!"
                    )

                states = X[column].isna()
                n_missing = round(len(X) * self.ratio_missing)
                samples_sizes = self.generate_hole_sizes(
                    X[[column]], nb_holes=n_missing
                )
                for sample in samples_sizes:
                    if sample > 0:
                        is_valid = (
                            ~(states | mask[column])
                            .rolling(sample + 2)
                            .max()
                            .fillna(1)
                            .astype(bool)
                        )
                        if not np.any(is_valid):
                            logger.warning("No place to introduce sampled hole!")
                            continue
                        i_hole = np.random.choice(np.where(is_valid)[0])
                        if mask[column].iloc[i_hole - sample + 1 : i_hole + 1].any():
                            print("overriding existing hole!")
                        mask[column].iloc[i_hole - sample : i_hole] = True

            list_masks.append(mask)
        return list_masks


class GroupedHoleGenerator(HoleGenerator):
    def __init__(
        self,
        n_splits: int,
        subset: Optional[List[str]] = None,
        ratio_missing: Optional[float] = 0.05,
        random_state: Optional[int] = 42,
        column_groups: Optional[List[str]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            subset=subset,
            random_state=random_state,
            ratio_missing=ratio_missing,
        )

        if self.column_groups is None:
            raise Exception("column_group is empty.")

        self.column_groups = column_groups

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

    def split(self, X: pd.DataFrame) -> List[pd.DataFrame]:

        if self.subset is None:
            self.subset = X.columns

        groups = self._create_groups(X)
        gss = GroupShuffleSplit(
            n_splits=self.n_splits,
            train_size=1 - self.ratio_missing,
            random_state=self.random_state,
        )
        mask_dfs = []
        for observed_indices, _ in gss.split(X=X, y=None, groups=groups):

            # create the boolean mask of missing values
            df_mask = pd.DataFrame(
                data=np.full((X.shape), True),
                columns=X.columns,
                index=X.index,
            )
            df_mask.iloc[observed_indices, self.subset] = False
            mask_dfs.append(df_mask)

        return mask_dfs
