import pytest
import numpy as np
import pandas as pd

from unittest.mock import patch, MagicMock
from qolmat.benchmark.comparator import Comparator

generator_holes_mock = MagicMock()
generator_holes_mock.split.return_value = [
    pd.DataFrame({"A": [False, False, True], "B": [True, False, False]})
]

comparator = Comparator(
    dict_models={},
    selected_columns=["A", "B"],
    generator_holes=generator_holes_mock,
    metrics=["mae", "mse"],
)

imputer_mock = MagicMock()
expected_get_errors = pd.Series(
    [1.0, 1.0, 1.0, 1.0],
    index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B"), ("mse", "A"), ("mse", "B")]),
)


@patch("qolmat.benchmark.metrics.get_metric")
def test_get_errors(mock_get_metric):
    df_origin = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]})
    df_imputed = pd.DataFrame({"A": [1, 2, 4], "B": [4, 5, 7]})
    df_mask = pd.DataFrame({"A": [False, False, True], "B": [False, False, True]})

    mock_get_metric.return_value = lambda df_origin, df_imputed, df_mask: pd.Series(
        [1.0, 1.0], index=["A", "B"]
    )
    errors = comparator.get_errors(df_origin, df_imputed, df_mask)
    pd.testing.assert_series_equal(errors, expected_get_errors)


@patch("qolmat.benchmark.hyperparameters.optimize", return_value=imputer_mock)
@patch(
    "qolmat.benchmark.comparator.Comparator.get_errors",
    return_value=expected_get_errors,
)
def test_evaluate_errors_sample(mock_get_errors, mock_optimize):
    errors_mean = comparator.evaluate_errors_sample(
        imputer_mock, pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, np.nan]})
    )
    expected_errors_mean = expected_get_errors
    pd.testing.assert_series_equal(errors_mean, expected_errors_mean)
    mock_optimize.assert_called_once()
    mock_get_errors.assert_called()


@patch(
    "qolmat.benchmark.comparator.Comparator.evaluate_errors_sample",
    return_value=expected_get_errors,
)
def test_compare(mock_evaluate_errors_sample):
    df_test = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    imputer1 = MagicMock(name="Imputer1")
    imputer2 = MagicMock(name="Imputer2")
    comparator.dict_imputers = {"imputer1": imputer1, "imputer2": imputer2}

    errors_imputer1 = pd.Series([0.1, 0.2], index=["mae", "mse"])
    errors_imputer2 = pd.Series([0.3, 0.4], index=["mae", "mse"])
    mock_evaluate_errors_sample.side_effect = [errors_imputer1, errors_imputer2]

    df_errors = comparator.compare(df_test)
    assert mock_evaluate_errors_sample.call_count == 2

    mock_evaluate_errors_sample.assert_any_call(imputer1, df_test, {}, "mse")
    mock_evaluate_errors_sample.assert_any_call(imputer2, df_test, {}, "mse")
    expected_df_errors = pd.DataFrame(
        {"imputer1": [0.1, 0.2], "imputer2": [0.3, 0.4]}, index=["mae", "mse"]
    )
    pd.testing.assert_frame_equal(df_errors, expected_df_errors)
