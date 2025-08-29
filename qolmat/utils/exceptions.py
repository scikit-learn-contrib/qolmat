"""Exceptions for qolmat package."""

from typing import Any, List, Tuple, Type


class PyTorchExtraNotInstalled(Exception):
    """Raise when pytorch extra is not installed."""

    def __init__(self):
        super().__init__(
            """Please install torch xx.xx.xx
        pip install qolmat[pytorch]"""
        )


class SignalTooShort(Exception):
    """Raise when the signal is too short."""

    def __init__(self, period: int, n_cols: int):
        super().__init__(
            f"""`period` must be smaller than the signals duration.
            `period`is {period} but the number of columns if {n_cols}!"""
        )


class NoMissingValue(Exception):
    """Raise an error when there is no missing value."""

    def __init__(self, subset_without_nans: List[str]):
        super().__init__(
            f"No missing value in the columns {subset_without_nans}! "
            "You need to pass the relevant column name in the subset argument!"
        )


class SubsetIsAString(Exception):
    """Raise an error when the subset is a string."""

    def __init__(self, subset: Any):
        super().__init__(
            f"Provided subset `{subset}` should be None or a list!"
        )


class NotDimension2(Exception):
    """Raise an error when the matrix is not of dim 2."""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__(
            f"Provided matrix is of shape {shape}, "
            "which is not of dimension 2!"
        )


class NotDataFrame(Exception):
    """Raise an error when the input is not a dataframe."""

    def __init__(self, X_type: Type[Any]):
        super().__init__(f"Input must be a dataframe, not a {X_type}")


class NotEnoughSamples(Exception):
    """Raise an error when there is no not enough samples."""

    def __init__(self, max_num_row: int, min_n_rows: int):
        super().__init__(
            f"Not enough valid patterns found. "
            f"Largest found pattern has {max_num_row} rows, when "
            f"they should have at least min_n_rows={min_n_rows}."
        )


class EstimatorNotDefined(Exception):
    """Raise an error when the estimator is not defined."""

    def __init__(self):
        super().__init__(
            "The underlying estimator should be defined beforehand!"
        )


class SingleSample(Exception):
    """Raise an error when there is a single sample."""

    def __init__(self):
        super().__init__(
            """This imputer cannot be fitted on a single sample!"""
        )


class IllConditioned(Exception):
    """Raise an error when the covariance matrix is ill-conditioned."""

    def __init__(self, min_sv: float, min_std: float):
        super().__init__(
            f"The covariance matrix is ill-conditioned, "
            "indicating high-colinearity: "
            "the smallest singular value of the data matrix is smaller "
            f"than the threshold min_std ({min_sv} < {min_std}). "
            f"Consider removing columns of decreasing the threshold."
        )


class TypeNotHandled(Exception):
    """Raise an error when the type is not handled."""

    def __init__(self, col: str, type_col: str):
        super().__init__(
            f"The column `{col}` is of type `{type_col}`, "
            "which is not handled!"
        )
