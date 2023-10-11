import numpy as np
import pandas as pd
import pytest

from typing import Any

from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from qolmat.benchmark import metrics
from qolmat.imputations import imputers, imputers_pytorch
from qolmat.imputations.diffusions import ddpms
from qolmat.utils.exceptions import PyTorchExtraNotInstalled

try:
    import torch
except ModuleNotFoundError:
    raise PyTorchExtraNotInstalled

df_incomplete = pd.DataFrame(
    {
        "col1": [np.nan, 15, np.nan, 23, 33],
        "col2": [69, 76, 74, 80, 78],
        "col3": [174, 166, 182, 177, np.nan],
        "col4": [9, 12, 11, 12, 8],
        "col5": [93, 75, np.nan, 12, np.nan],
    },
    index=pd.date_range("2023-04-17", periods=5, freq="D"),
)
df_incomplete.index = df_incomplete.index.set_names("datetime")


@pytest.mark.parametrize("df", [df_incomplete])
def test_ImputerDiffusion_fit_transform(df: pd.DataFrame) -> None:
    expected = pd.Series(
        {
            "col1": False,
            "col2": False,
            "col3": False,
            "col4": False,
            "col5": False,
        }
    )

    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    imputer = imputers_pytorch.ImputerDiffusion(
        model=model, batch_size=2, epochs=2, x_valid=df, print_valid=True
    )

    df_imputed = imputer.fit_transform(df)
    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)
    np.testing.assert_array_equal(np.isnan(df_imputed).any(), expected)

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    imputer = imputers_pytorch.ImputerDiffusion(
        model=model,
        batch_size=2,
        epochs=2,
        x_valid=df,
        print_valid=True,
        index_datetime="datetime",
    )

    df_imputed = imputer.fit_transform(df)
    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)
    np.testing.assert_array_equal(np.isnan(df_imputed).any(), expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_fit(df: pd.DataFrame) -> None:
    expected = pd.Series(
        {
            "col1": False,
            "col2": False,
            "col3": False,
            "col4": False,
            "col5": False,
        }
    )

    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    model = model.fit(df, batch_size=2, epochs=2, x_valid=df, print_valid=False)

    df_imputed = model.predict(df)

    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)
    np.testing.assert_array_equal(np.isnan(df_imputed).any(), expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_process_data(df: pd.DataFrame) -> None:

    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    arr_processed, arr_mask, _ = model._process_data(df, is_training=True)

    np.testing.assert_array_equal(df.shape, arr_processed.shape)
    np.testing.assert_array_equal(arr_mask, df.notna().values)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_process_reversely_data(df: pd.DataFrame) -> None:

    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    model = model.fit(df, batch_size=2, epochs=2, x_valid=df, print_valid=False)

    arr_processed, arr_mask, list_indices = model._process_data(df, is_training=False)
    df_imputed = model._process_reversely_data(arr_processed, df, list_indices)

    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_q_sample(df: pd.DataFrame) -> None:

    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    model = model.fit(df, batch_size=2, epochs=2, x_valid=df, print_valid=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ts_data_noised, ts_noise = model._q_sample(
        x=torch.ones(2, 5, dtype=torch.float).to(device),
        t=torch.ones(2, 1, dtype=torch.long).to(device),
    )

    np.testing.assert_array_equal(ts_data_noised.size(), (2, 5))
    np.testing.assert_array_equal(ts_noise.size(), (2, 5))


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_eval(df: pd.DataFrame) -> None:
    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64, is_clip=True)
    model = model.fit(
        df,
        batch_size=2,
        epochs=2,
        x_valid=df,
        print_valid=False,
        metrics_valid=(
            metrics.mean_absolute_error,
            metrics.dist_wasserstein,
        ),
    )

    scores = model._eval(
        df.fillna(df.mean()).values,
        df.notna().values,
        df.fillna(df.mean()),
        df.notna(),
        list(df.index),
    )

    np.testing.assert_array_equal(list(scores.keys()), ["mean_absolute_error", "dist_wasserstein"])


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_impute(df: pd.DataFrame) -> None:
    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    model = model.fit(
        df,
        batch_size=2,
        epochs=2,
        x_valid=df,
        print_valid=False,
        metrics_valid=(
            metrics.mean_absolute_error,
            metrics.dist_wasserstein,
        ),
    )

    arr_imputed = model._impute(df.fillna(df.mean()).values, df.notna().values)

    np.testing.assert_array_equal(df.shape, arr_imputed.shape)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TabDDPM_predict(df: pd.DataFrame) -> None:
    expected = pd.Series(
        {
            "col1": False,
            "col2": False,
            "col3": False,
            "col4": False,
            "col5": False,
        }
    )

    model = ddpms.TabDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64, is_clip=True)
    model = model.fit(df, batch_size=2, epochs=2, x_valid=df, print_valid=False)

    df_imputed = model.predict(df)

    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)
    np.testing.assert_array_equal(np.isnan(df_imputed).any(), expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TsDDPM_fit(df: pd.DataFrame) -> None:
    expected = pd.Series(
        {
            "col1": False,
            "col2": False,
            "col3": False,
            "col4": False,
            "col5": False,
        }
    )

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    model = model.fit(
        df, batch_size=2, epochs=2, x_valid=df, print_valid=False, index_datetime="datetime"
    )

    df_imputed = model.predict(df)

    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)
    np.testing.assert_array_equal(np.isnan(df_imputed).any(), expected)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TsDDPM_process_data(df: pd.DataFrame) -> None:

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64, is_rolling=False)
    model = model.fit(
        df, batch_size=2, epochs=2, x_valid=df, print_valid=False, index_datetime="datetime"
    )

    arr_processed, arr_mask, _ = model._process_data(df, is_training=True)

    np.testing.assert_array_equal(arr_processed.shape, [5, 1, 5])
    np.testing.assert_array_equal(arr_mask.shape, [5, 1, 5])

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64, is_rolling=True)
    model = model.fit(
        df, batch_size=2, epochs=2, x_valid=df, print_valid=False, index_datetime="datetime"
    )

    arr_processed, arr_mask, _ = model._process_data(df, is_training=True)

    np.testing.assert_array_equal(arr_processed.shape, [5, 1, 5])
    np.testing.assert_array_equal(arr_mask.shape, [5, 1, 5])


@pytest.mark.parametrize("df", [df_incomplete])
def test_TsDDPM_process_reversely_data(df: pd.DataFrame) -> None:

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64, is_rolling=False)
    model = model.fit(
        df, batch_size=2, epochs=2, x_valid=df, print_valid=False, index_datetime="datetime"
    )

    arr_processed, arr_mask, list_indices = model._process_data(df, is_training=False)
    df_imputed = model._process_reversely_data(arr_processed, df, list_indices)

    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64, is_rolling=True)
    model = model.fit(
        df, batch_size=2, epochs=2, x_valid=df, print_valid=False, index_datetime="datetime"
    )

    arr_processed, arr_mask, list_indices = model._process_data(df, is_training=False)
    df_imputed = model._process_reversely_data(arr_processed, df, list_indices)

    np.testing.assert_array_equal(df.shape, df_imputed.shape)
    np.testing.assert_array_equal(df.index, df_imputed.index)
    np.testing.assert_array_equal(df.columns, df_imputed.columns)


@pytest.mark.parametrize("df", [df_incomplete])
def test_TsDDPM_q_sample(df: pd.DataFrame) -> None:

    model = ddpms.TsDDPM(num_noise_steps=10, num_blocks=1, dim_embedding=64)
    model = model.fit(
        df, batch_size=2, epochs=2, x_valid=df, print_valid=False, index_datetime="datetime"
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ts_data_noised, ts_noise = model._q_sample(
        x=torch.ones(2, 1, 5, dtype=torch.float).to(device),
        t=torch.ones(2, 1, 1, dtype=torch.long).to(device),
    )

    np.testing.assert_array_equal(ts_data_noised.size(), (2, 1, 5))
    np.testing.assert_array_equal(ts_noise.size(), (2, 1, 5))


@parametrize_with_checks(
    [
        imputers_pytorch.ImputerDiffusion(model=ddpms.TabDDPM(), batch_size=1, epochs=1),
    ]
)
def test_sklearn_compatible_estimator(estimator: imputers._Imputer, check: Any) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    check(estimator)
