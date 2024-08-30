import pandas as pd
import pytest

from qolmat.utils.exceptions import TypeNotHandled
from qolmat.utils.input_check import check_pd_df_dtypes


@pytest.fixture
def multitypes_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True],
        'datetime_col': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
    })


@pytest.fixture
def supported_multitypes_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })


def test__check_pd_df_dtypes_raise_error(multitypes_dataframe):
    with pytest.raises(TypeNotHandled):
        check_pd_df_dtypes(
            multitypes_dataframe, [
                pd.api.types.is_numeric_dtype,
                pd.api.types.is_string_dtype,
                pd.api.types.is_bool_dtype
            ]
        )


def test__check_pd_df_dtypes(supported_multitypes_dataframe):
    check_pd_df_dtypes(
        supported_multitypes_dataframe,
        [
            pd.api.types.is_numeric_dtype,
            pd.api.types.is_string_dtype,
            pd.api.types.is_bool_dtype
        ]
    )
