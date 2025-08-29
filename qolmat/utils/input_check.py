"""Util file for input checks."""

import pandas as pd

from qolmat.utils.exceptions import TypeNotHandled


def check_pd_df_dtypes(df: pd.DataFrame, allowed_types: list):
    """Validate that the columns of the DataFrame have allowed data types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns' data types are to be checked.
    allowed_types : list
        List which contains the allowed data types.

    Raises
    ------
    TypeNotHandled
        If any column has a data type that is not numeric, string, or boolean.

    """

    def is_allowed_type(dtype):
        return any(check(dtype) for check in allowed_types)

    invalid_columns = [
        (col, dtype)
        for col, dtype in df.dtypes.items()
        if not is_allowed_type(dtype)
    ]
    if invalid_columns:
        for column_name, dtype in invalid_columns:
            raise TypeNotHandled(col=str(column_name), type_col=dtype)
