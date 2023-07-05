from typing import Any, List, Tuple
from numpy.typing import NDArray


class KerasExtraNotInstalled(Exception):
    def __init__(self):
        super().__init__(
            """Please install keras xx.xx.xx
        pip install qolmat[keras]"""
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
