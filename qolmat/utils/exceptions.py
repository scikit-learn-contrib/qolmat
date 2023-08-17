from typing import Any, List, Tuple, Type


class PyTorchExtraNotInstalled(Exception):
    def __init__(self):
        super().__init__(
            """Please install torch xx.xx.xx
        pip install qolmat[pytorch]"""
        )


class SignalTooShort(Exception):
    def __init__(self, period: int, n_cols: int):
        super().__init__(
            f"""`period` must be smaller than the signals duration.
            `period`is {period} but the number of columns if {n_cols}!"""
        )


class NoMissingValue(Exception):
    def __init__(self, subset_without_nans: List[str]):
        super().__init__(
            f"No missing value in the columns {subset_without_nans}! "
            "You need to pass the relevant column name in the subset argument!"
        )


class SubsetIsAString(Exception):
    def __init__(self, subset: Any):
        super().__init__(f"Provided subset `{subset}` should be None or a list!")


class NotDimension2(Exception):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__(f"Provided matrix is of shape {shape}, which is not of dimension 2!")


class NotDataFrame(Exception):
    def __init__(self, X_type: Type[Any]):
        super().__init__(f"Input musr be a dataframe, not a {X_type}")


class NotEnoughSamples(Exception):
    def __init__(self, max_num_row: int, min_n_rows: int):
        super().__init__(
            f"Not enough valid patterns found. Largest found pattern has {max_num_row} rows, when "
            f"they should have at least min_n_rows={min_n_rows}."
        )


class EstimatorNotDefined(Exception):
    def __init__(self):
        super().__init__("The underlying estimator should be defined beforehand!")


class SingleSample(Exception):
    def __init__(self):
        super().__init__("""This imputer cannot be fitted on a single sample!""")
