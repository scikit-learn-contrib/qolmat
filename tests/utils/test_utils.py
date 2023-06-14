import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from qolmat.utils import utils
from pytest_mock.plugin import MockerFixture
from io import StringIO

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.mark.parametrize("df", [df])
def test_utils_utils_display_bar_table(df: pd.DataFrame, mocker: MockerFixture) -> None:
    mock_save = mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.show")
    utils.display_bar_table(data=df, ylabel="Counts", path="output/barplot")
    y_label_text = plt.gca().get_ylabel()
    assert mock_save.call_count == 1
    assert plt.gcf() is not None
    assert plt.gca() is not None
    assert y_label_text == "Counts"


@pytest.mark.parametrize("iteration, total", [(1, 1)])
def test_utils_utils_display_progress_bar(iteration: int, total: int, capsys) -> None:
    captured_output = StringIO()
    sys.stdout = captured_output
    utils.progress_bar(
        iteration, total, prefix="Progress", suffix="Complete", decimals=1, length=2, fill="█"
    )
    captured_output.seek(0)
    output = captured_output.read().strip()
    sys.stdout = sys.__stdout__

    output_expected = "Progress |██| 100.0% Complete"
    assert output == output_expected


@pytest.mark.parametrize("values, lag_max", [(pd.Series([1, 2, 3, 4, 5]), 3)])
def test_utils_utils_acf(values, lag_max):
    result = utils.acf(values, lag_max)
    result_expected = pd.Series([1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result, result_expected, atol=0.001)
