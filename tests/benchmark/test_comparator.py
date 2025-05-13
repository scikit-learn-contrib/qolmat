"""Tests for Comparator class.

Class:
    TestComparator: group tests for Comparator class.

"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from qolmat.benchmark.comparator import Comparator
from qolmat.benchmark.missing_patterns import _HoleGenerator


@pytest.fixture
def generator_holes_mock(mocker: MockerFixture) -> _HoleGenerator:
    mock = mocker.create_autospec(_HoleGenerator, instance=True)
    mock.split.return_value = [
        pd.DataFrame({"A": [False, False, True], "B": [True, False, False]})
    ]
    mock.n_splits = 1
    mock.subset = ["A", "B"]
    mock.ratio_masked = 0.3
    mock.random_state = 42
    mock.groups = None

    return mock


@pytest.fixture
def comparator(generator_holes_mock: _HoleGenerator) -> Comparator:
    return Comparator(
        dict_models={},
        selected_columns=["A", "B"],
        generator_holes=generator_holes_mock,
        metrics=["mae", "mse"],
    )


@pytest.fixture
def expected_get_errors() -> pd.Series:
    return pd.Series(
        [1.0, 1.0, 1.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [("mae", "A"), ("mae", "B"), ("mse", "A"), ("mse", "B")]
        ),
    )


@pytest.fixture
def df_origin() -> pd.DataFrame:
    return pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]})


@pytest.fixture
def df_imputed() -> pd.DataFrame:
    return pd.DataFrame({"A": [1, 2, 4], "B": [4, 5, 7]})


@pytest.fixture
def df_mask() -> pd.DataFrame:
    return pd.DataFrame({"A": [False, False, True], "B": [False, False, True]})


@pytest.fixture
def imputers_mock(mocker: MockerFixture) -> Dict[str, Any]:
    imputer_mock = mocker.MagicMock()
    imputer_mock.fit_transform.return_value = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6]}
    )
    return {"imputer_1": imputer_mock}


@pytest.fixture
def config_opti_mock() -> Dict[str, Dict[str, Any]]:
    return {"imputer_1": {"param_1": "value"}}


@pytest.fixture
def all_masks() -> List[pd.DataFrame]:
    return [
        pd.DataFrame({"A": [False, True, False], "B": [False, False, True]}),
        pd.DataFrame({"A": [True, False, False], "B": [False, True, False]}),
    ]


class TestComparator:
    """Group tests for Comparator class."""

    def test_get_errors(
        self,
        mocker: MockerFixture,
        comparator: Comparator,
        expected_get_errors: pd.Series,
        df_origin: pd.DataFrame,
        df_imputed: pd.DataFrame,
        df_mask: pd.DataFrame,
    ) -> None:
        """Test the get_errors method."""
        mock_get_metric = mocker.patch("qolmat.benchmark.metrics.get_metric")
        mock_get_metric.return_value = (
            lambda df_origin, df_imputed, df_mask: pd.Series(
                [1.0, 1.0], index=["A", "B"]
            )
        )

        errors = comparator.get_errors(df_origin, df_imputed, df_mask)

        pd.testing.assert_series_equal(errors, expected_get_errors)

    def test_process_split(
        self,
        mocker,
        comparator,
        imputers_mock,
        config_opti_mock,
        df_origin,
        df_mask,
    ):
        """Test the process_split method."""
        comparator.dict_imputers = imputers_mock
        comparator.dict_config_opti = config_opti_mock
        comparator.metric_optim = "mae"
        comparator.max_evals = 100
        comparator.verbose = False

        mock_optimize = mocker.patch(
            "qolmat.benchmark.comparator.hyperparameters.optimize"
        )
        mock_optimize.return_value = imputers_mock["imputer_1"]
        split_data = (0, df_mask, df_origin)
        df_with_holes = df_origin.copy()
        df_with_holes[df_mask] = np.nan

        result = comparator.process_split(split_data)

        assert isinstance(result, pd.DataFrame)
        assert "imputer_1" in result.columns
        assert {"mae", "mse"} == set(
            result.index.get_level_values(0)
        ), "Index level 0 should be 'mae' and 'mse'"
        assert {"A", "B"} == set(
            result.index.get_level_values(1)
        ), "Index level 1 should be 'A' and 'B'"

        mock_optimize.assert_called_once_with(
            imputers_mock["imputer_1"],
            df_origin,
            comparator.generator_holes,
            comparator.metric_optim,
            config_opti_mock["imputer_1"],
            max_evals=comparator.max_evals,
            verbose=comparator.verbose,
        )
        args, _ = imputers_mock["imputer_1"].fit_transform.call_args
        pd.testing.assert_frame_equal(args[0], df_with_holes)

    def test_process_imputer(
        self,
        mocker: MockerFixture,
        comparator: Comparator,
        imputers_mock: Dict[str, Any],
        config_opti_mock: Dict[str, Dict[str, Any]],
        all_masks: List[pd.DataFrame],
        df_origin: pd.DataFrame,
    ) -> None:
        """Test the process_imputer method."""
        comparator.dict_imputers = imputers_mock
        comparator.dict_config_opti = config_opti_mock
        comparator.metric_optim = "mae"
        comparator.max_evals = 100
        comparator.verbose = False
        mock_optimize = mocker.patch(
            "qolmat.benchmark.comparator.hyperparameters.optimize"
        )
        mock_optimize.return_value = imputers_mock["imputer_1"]
        mock_get_errors = mocker.patch.object(comparator, "get_errors")
        mock_get_errors.side_effect = [
            pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
            ),
            pd.Series(
                [1.5, 2.5],
                index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
            ),
        ]
        expected_result = pd.Series(
            [1.25, 2.25],
            index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
        )

        imputer_data = (
            "imputer_1",
            imputers_mock["imputer_1"],
            all_masks,
            df_origin,
        )
        imputer_name, result = comparator.process_imputer(imputer_data)

        assert imputer_name == "imputer_1"
        pd.testing.assert_series_equal(result, expected_result)
        mock_optimize.assert_called_once_with(
            imputers_mock["imputer_1"],
            df_origin,
            comparator.generator_holes,
            comparator.metric_optim,
            config_opti_mock["imputer_1"],
            max_evals=comparator.max_evals,
            verbose=comparator.verbose,
        )
        assert imputers_mock["imputer_1"].fit_transform.call_count == len(
            all_masks
        )
        assert mock_get_errors.call_count == len(all_masks)

    def test_compare_parallel_splits(
        self,
        mocker: MockerFixture,
        comparator: Comparator,
        df_origin: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test the compare method with parallel splits."""
        mock_split = mocker.patch.object(comparator.generator_holes, "split")
        mock_split.return_value = [
            pd.DataFrame(
                {"A": [False, True, False], "B": [True, False, True]}
            ),
            pd.DataFrame(
                {"A": [True, False, True], "B": [False, True, False]}
            ),
        ]
        mock_process_split = mocker.patch.object(comparator, "process_split")
        mock_process_split.side_effect = [
            pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
            ),
            pd.Series(
                [1.5, 2.5],
                index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
            ),
        ]
        mock_get_optimal_n_jobs = mocker.patch.object(
            comparator, "get_optimal_n_jobs"
        )
        mock_get_optimal_n_jobs.return_value = 1

        expected_result = pd.Series(
            [1.25, 2.25],
            index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
        )

        with caplog.at_level(logging.INFO):
            result = comparator.compare(df_origin, parallel_over="splits")

        pd.testing.assert_series_equal(result, expected_result)
        assert mock_process_split.call_count == 2
        assert mock_get_optimal_n_jobs.call_count == 1
        assert "Starting comparison for" in caplog.text
        assert "Comparison successfully terminated." in caplog.text

    def test_compare_sequential_splits(
        self,
        mocker: MockerFixture,
        comparator: Comparator,
        df_origin: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test the compare method with sequential splits."""
        mock_split = mocker.patch.object(comparator.generator_holes, "split")
        mock_split.return_value = [
            pd.DataFrame(
                {"A": [False, True, False], "B": [True, False, True]}
            ),
            pd.DataFrame(
                {"A": [True, False, True], "B": [False, True, False]}
            ),
        ]
        mock_process_split = mocker.patch.object(comparator, "process_split")
        mock_process_split.side_effect = [
            pd.Series(
                [1.0, 2.0],
                index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
            ),
            pd.Series(
                [1.5, 2.5],
                index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
            ),
        ]

        expected_result = pd.Series(
            [1.25, 2.25],
            index=pd.MultiIndex.from_tuples([("mae", "A"), ("mae", "B")]),
        )

        with caplog.at_level(logging.INFO):
            result = comparator.compare(
                df_origin, use_parallel=False, parallel_over="splits"
            )

        pd.testing.assert_series_equal(result, expected_result)
        assert mock_process_split.call_count == 2
        assert "Starting comparison for" in caplog.text
        assert "Comparison successfully terminated." in caplog.text

    def test_compare_parallel_imputers(
        self,
        mocker: MockerFixture,
        comparator: Comparator,
        df_origin: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test the compare method with parallel imputers."""
        mock_split = mocker.patch.object(comparator.generator_holes, "split")
        mock_split.return_value = [
            pd.DataFrame(
                {"A": [False, True, False], "B": [True, False, True]}
            ),
            pd.DataFrame(
                {"A": [True, False, True], "B": [False, True, False]}
            ),
        ]
        comparator.dict_imputers = {
            "imputer_1": mocker.Mock(),
            "imputer_2": mocker.Mock(),
        }
        mock_process_imputer = mocker.patch.object(
            comparator, "process_imputer"
        )
        mock_process_imputer.side_effect = [
            (
                "imputer_1",
                pd.DataFrame(
                    {"A": [1.0, 2.0], "B": [3.0, 4.0]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
            ),
            (
                "imputer_2",
                pd.DataFrame(
                    {"A": [1.5, 2.5], "B": [3.5, 4.5]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
            ),
        ]
        mock_get_optimal_n_jobs = mocker.patch.object(
            comparator, "get_optimal_n_jobs"
        )
        mock_get_optimal_n_jobs.return_value = 1

        expected_result = pd.concat(
            {
                "imputer_1": pd.DataFrame(
                    {"A": [1.0, 2.0], "B": [3.0, 4.0]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
                "imputer_2": pd.DataFrame(
                    {"A": [1.5, 2.5], "B": [3.5, 4.5]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
            },
            axis=1,
        )

        with caplog.at_level(logging.INFO):
            result = comparator.compare(
                df_origin, use_parallel=True, parallel_over="imputers"
            )

        pd.testing.assert_frame_equal(result, expected_result)
        assert mock_process_imputer.call_count == 2
        assert mock_get_optimal_n_jobs.call_count == 1
        assert "Starting comparison for" in caplog.text
        assert "Comparison successfully terminated." in caplog.text

    def test_compare_sequential_imputers(
        self,
        mocker: MockerFixture,
        comparator: Comparator,
        df_origin: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test the compare method with sequential imputers."""
        mock_split = mocker.patch.object(comparator.generator_holes, "split")
        mock_split.return_value = [
            pd.DataFrame(
                {"A": [False, True, False], "B": [True, False, True]}
            ),
            pd.DataFrame(
                {"A": [True, False, True], "B": [False, True, False]}
            ),
        ]
        comparator.dict_imputers = {
            "imputer_1": mocker.Mock(),
            "imputer_2": mocker.Mock(),
        }
        mock_process_imputer = mocker.patch.object(
            comparator, "process_imputer"
        )
        mock_process_imputer.side_effect = [
            (
                "imputer_1",
                pd.DataFrame(
                    {"A": [1.0, 2.0], "B": [3.0, 4.0]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
            ),
            (
                "imputer_2",
                pd.DataFrame(
                    {"A": [1.5, 2.5], "B": [3.5, 4.5]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
            ),
        ]

        expected_result = pd.concat(
            {
                "imputer_1": pd.DataFrame(
                    {"A": [1.0, 2.0], "B": [3.0, 4.0]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
                "imputer_2": pd.DataFrame(
                    {"A": [1.5, 2.5], "B": [3.5, 4.5]},
                    index=pd.MultiIndex.from_tuples(
                        [("mae", "A"), ("mae", "B")]
                    ),
                ),
            },
            axis=1,
        )

        with caplog.at_level(logging.INFO):
            result = comparator.compare(
                df_origin, use_parallel=False, parallel_over="imputers"
            )

        pd.testing.assert_frame_equal(result, expected_result)
        assert mock_process_imputer.call_count == 2
        assert "Starting comparison for" in caplog.text
        assert "Comparison successfully terminated." in caplog.text

    def test_get_optimal_n_jobs_with_specified_n_jobs(self) -> None:
        """Test when n_jobs is specified."""
        split_data = [1, 2, 3, 4]
        n_jobs = 2

        result = Comparator.get_optimal_n_jobs(split_data, n_jobs=n_jobs)

        assert (
            result == n_jobs
        ), f"Expected n_jobs to be {n_jobs}, but got {result}"

    def test_get_optimal_n_jobs_with_default_n_jobs(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test when n_jobs is not specified."""
        split_data = [1, 2, 3, 4]

        mocker.patch("multiprocessing.cpu_count", return_value=8)

        result = Comparator.get_optimal_n_jobs(split_data, n_jobs=-1)
        assert result == len(
            split_data
        ), f"Expected {len(split_data)}, but got {result}"

    def test_get_optimal_n_jobs_with_large_cpu_count(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test when number of CPUs is greater than the len of split_data."""
        split_data = [1, 2]  # Seulement 2 tâches

        mocker.patch("multiprocessing.cpu_count", return_value=16)

        result = Comparator.get_optimal_n_jobs(split_data, n_jobs=-1)
        assert result == len(
            split_data
        ), f"Expected {len(split_data)}, but got {result}"
