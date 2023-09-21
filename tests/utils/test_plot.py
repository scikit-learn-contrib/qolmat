from typing import Any, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from qolmat.utils import plot
from pytest_mock.plugin import MockerFixture

plt.switch_backend("Agg")

np.random.seed(42)
matrice1 = np.random.rand(3, 3)
matrice2 = np.random.rand(3, 3)
matrice3 = np.random.rand(3, 3)
list_matrices = [matrice1, matrice2, matrice3]

signal1 = [1, -1, 1, 0, 1, 1 / 2, -1 / 2, 0, 1 / 2]
signal2 = [0, -1 / 2, 1 / 2, 1, -1, 1 / 2, 1 / 2, 1, -1 / 2]
list_signals = [signal1, signal2]

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

M = np.random.rand(100, 100)
A = np.random.rand(100, 100)
E = scipy.sparse.csr_matrix(np.random.rand(100, 10))

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
df1 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
df2 = pd.DataFrame({"x": [2, 3, 4], "y": [5, 6, 7]})
dict_df_imputed = {
    "Imputer1": pd.DataFrame({"A": [2, 3, np.nan], "B": [5, np.nan, 7], "C": [np.nan, 8, 9]})
}


@pytest.mark.parametrize("list_matrices", [list_matrices])
def test_utils_plot_plot_matrices(list_matrices: List[np.ndarray], mocker: MockerFixture) -> None:
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.show")
    plot.plot_matrices(list_matrices=list_matrices, title="title")
    assert len(plt.gcf().get_axes()) > 0
    assert plt.savefig.call_count == 1
    plt.close("all")


@pytest.mark.parametrize("list_signals", [list_signals])
def test_utils_plot_plot_signal(list_signals: List[List[Any]], mocker: MockerFixture) -> None:
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.show")
    plot.plot_signal(list_signals=list_signals, ylabel="ylabel", title="title")
    assert len(plt.gcf().get_axes()) > 0
    assert plt.savefig.call_count == 1
    plt.close("all")


@pytest.mark.parametrize("M, A, E, index_array, dims", [(M, A, E, [0, 1, 2], (10, 10))])
def test__utils_plot_plot_images(
    M: np.ndarray,
    A: np.ndarray,
    E: np.ndarray,
    index_array: List[int],
    dims: Tuple[int, int],
    mocker: MockerFixture,
):
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.show")
    plot.plot_images(M, A, E, index_array, dims, filename="filename")
    assert len(plt.gcf().get_axes()) > 0
    assert plt.savefig.call_count == 1
    plt.close("all")


@pytest.mark.parametrize("X", [X])
def test_utils_plot_make_ellipses(X: np.ndarray, mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    ax = plt.gca()
    plot.make_ellipses(X[1], X[2], ax, color="blue")
    assert len(plt.gcf().get_axes()) > 0
    plt.close("all")


@pytest.mark.parametrize("df1,df2", [(df1, df2)])
def test_utils_plot_compare_covariances(
    df1: pd.DataFrame, df2: pd.DataFrame, mocker: MockerFixture
):
    mocker.patch("matplotlib.pyplot.show")
    ax = plt.gca()
    plot.compare_covariances(df1, df2, "x", "y", ax)
    assert len(plt.gcf().get_axes()) > 0
    plt.close("all")


@pytest.mark.parametrize("df", [df])
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_utils_plot_multibar(df: pd.DataFrame, orientation: str, mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    plot.multibar(df, orientation=orientation)
    assert len(plt.gcf().get_axes()) > 0
    plt.close("all")


@pytest.mark.parametrize("df", [df])
def test_utils_plot_plot_imputations(df: pd.DataFrame, mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    plot.plot_imputations(df, dict_df_imputed)
    assert len(plt.gcf().get_axes()) > 0
    plt.close("all")
