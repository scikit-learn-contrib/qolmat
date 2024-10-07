"""===============================================
Tutorial for imputers based on diffusion models
===============================================

In this tutorial, we show how to use :class:`~qolmat.imputations.diffusions.ddpms.TabDDPM`
and :class:`~qolmat.imputations.diffusions.ddpms.TsDDPM` classes.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qolmat.benchmark import comparator, missing_patterns
from qolmat.imputations.diffusions.ddpms import TabDDPM, TsDDPM
from qolmat.imputations.imputers_pytorch import ImputerDiffusion
from qolmat.utils import data

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# %%
# 1. Time-series data
# ---------------------------------------------------------------
# We use the public Beijing Multi-Site Air-Quality Data Set.
# It consists in hourly air pollutants data from 12 chinese nationally-controlled air-quality
# monitoring sites. The original data from which the features were extracted comes from
# https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip.
# For this tutorial, we only use a small subset of this data
# 1000 rows and 2 features (TEMP, PRES).

df_data = data.get_data_corrupted("Beijing")
df_data = df_data[["TEMP", "PRES"]].iloc[:1000]
df_data.index = df_data.index.set_levels(
    [df_data.index.levels[0], pd.to_datetime(df_data.index.levels[1])]
)

logging.info(f"Number of nan at each column: {df_data.isna().sum()}")

# %%
# 2. Hyperparameters for the wapper ImputerDiffusion
# ---------------------------------------------------------------
# We use the wapper :class:`~qolmat.imputations.imputers_pytorch.ImputerDiffusion` for our
# diffusion models (e.g., :class:`~qolmat.imputations.diffusions.ddpms.TabDDPM`,
# :class:`~qolmat.imputations.diffusions.ddpms.TsDDPM`). The most important hyperparameter
# is ``model`` where we select a diffusion base model for the task of imputation
# (e.g., ``model=TabDDPM()``).
# Other hyperparams are for training the selected diffusion model.
#
# * ``cols_imputed``: list of columns that need to be imputed. Recall that we train the model on
#   incomplete data by using the self-supervised learning method. We can set which columns to be
#   masked during training. Its defaut value is ``None``.
#
# * ``epochs`` : a number of iterations, its defaut value ``epochs=10``. In practice, we should
#   set a larger number of epochs e.g., ``epochs=100``.
#
# * ``batch_size`` : a size of batch, its defaut value ``batch_size=100``.
#
# The following hyperparams are for validation:
#
# * ``x_valid``: a validation set.
#
# * ``metrics_valid``: a list validation metrics (see all [metrics](imputers.html). Its default
#   value ``metrics_valid=(metrics.mean_absolute_error, metrics.dist_wasserstein,)``
#
# * ``print_valid``: a boolean to display/hide a training progress (including epoch_loss,
#   remaining training duration and performance scores computed by the metrics above).

df_data_valid = df_data.iloc[:500]

tabddpm = ImputerDiffusion(
    model=TabDDPM(),
    epochs=10,
    batch_size=100,
    x_valid=df_data_valid,
    print_valid=True,
)
tabddpm = tabddpm.fit(df_data)

# %%
# We can see the architecture of the TabDDPM with ``get_summary_architecture()``

logging.info(tabddpm.get_summary_architecture())

# %%
# We also get the summary of the training progress with ``get_summary_training()``

summary = tabddpm.get_summary_training()

logging.info(f"Performance metrics: {list(summary.keys())}")


metric = "mean_absolute_error"
metric_scores = summary[metric]

fig, ax = plt.subplots()
ax.plot(range(len(metric_scores)), metric_scores)
ax.set_xlabel("Epoch")
ax.set_ylabel(metric)

plt.show()


# %%
# We display the imputations for the variable TEMP.

df_imputed = tabddpm.transform(df_data)

station = df_data.index.get_level_values("station")[0]
col = "TEMP"

values_orig = df_data.loc[station, col]
values_imp = df_imputed.loc[station, col].copy()

fig, ax = plt.subplots(figsize=(10, 3))
plt.plot(values_orig, ".", color="black", label="original")

values_imp[values_orig.notna()] = np.nan

plt.plot(values_imp, ".", color="blue", label="TabDDPM")
plt.ylabel(col, fontsize=10)
plt.legend(loc=[1.01, 0], fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=10)
plt.show()

# %%
# 3. Hyperparameters for TabDDPM
# ---------------------------------------------------------------
# :class:`~qolmat.imputations.diffusions.ddpms.TabDDPM` is a diffusion model based on
# Denoising Diffusion Probabilistic Models [1] for imputing tabular data. Several important
# hyperparameters are
#
# * ``num_noise_steps``: the number of step in the forward/reverse process.
#   It is T in the equation 1 of [1]. Its default value ``num_noise_steps=50``.
#   Note that a larger value can improve imputation quality but also increases inference time.
#
# * ``beta_start`` and ``beta_end``: the minimum and the maximum value
#   for the linear variance schedule (equation 2 of [1]).
#   Their default values ``beta_start=1e-4``, ``beta_end=0.02``
#
# * ``num_sampling``: for each missing value, the model generates n imputation variants.
#   The mean value of these variants is returned.
#   Based on our experiments, a large n (n > 5) often improves reconstruction scores (e.g., MAE).
#   Its default value ``num_sampling=1``.
#
# * ``ratio_nan=0.1``: in the self-supervised learning method, we need to randomly mask partial
#   observed data based on this ratio of missing values.
#
# Other hyperparams for building this deep learning model are
#
# * ``lr``: learning rate (``float = 0.001``)
#
# * ``num_blocks``: number of residual blocks (``int = 1)``
#
# * ``dim_embedding``: dimension of hidden layers in residual blocks (``int = 128``)
#
# Let see an example below. We can observe that a large ``num_sampling`` generally improves
# reconstruction errors (mae) but increases distribution distance (kl_columnwise).

dict_imputers = {
    "num_sampling=5": ImputerDiffusion(
        model=TabDDPM(num_sampling=5), epochs=10, batch_size=100
    ),
    "num_sampling=10": ImputerDiffusion(
        model=TabDDPM(num_sampling=10), epochs=10, batch_size=100
    ),
}

comparison = comparator.Comparator(
    dict_imputers,
    selected_columns=df_data.columns,
    generator_holes=missing_patterns.UniformHoleGenerator(n_splits=2),
    metrics=["mae", "kl_columnwise"],
)
results = comparison.compare(df_data)

results.groupby(axis=0, level=0).mean().groupby(axis=0, level=0).mean()

# %%
# 4. Hyperparameters for TsDDPM
# ---------------------------------------------------------------
# :class:`~qolmat.imputations.diffusions.ddpms.TsDDPM` is built on top of
# :class:`~qolmat.imputations.diffusions.ddpms.TabDDPM` to capture time-based relationships
# between data points in a dataset.
#
# Two important hyperparameters for processing time-series data are ``index_datetime``
# and ``freq_str``.
# E.g., ``ImputerDiffusion(model=TabDDPM(), index_datetime='datetime', freq_str='1D')``,
#
# * ``index_datetime``: the column name of datetime in index. It must be a pandas datetime object.
#
# * ``freq_str``: the time-series frequency for splitting data into a list of chunks (each chunk
#   has the same number of rows). These chunks are fetched up in batches.
#   A large frequency e.g., ``6M``, ``1Y`` can cause the out of memory.
#   Its default value ``freq_str: str = "1D"``. Time series frequencies can be found in this
#   `link <https://pandas.pydata.org/pandas-docs/
#   stable/user_guide/timeseries.html#offset-aliases>`_
#
# For TsDDPM, we have two options for splitting data:
#
# * ``is_rolling=False`` (default value): the data is splited by using
#   pandas.DataFrame.resample(rule=freq_str). There is no duplication of row between chunks,
#   leading a smaller number of chunks than the number of rows in the original data.
#
# * ``is_rolling=True``: the data is splited by using pandas.DataFrame.rolling(window=freq_str).
#   The number of chunks is also the number of rows in the original data.
#   Note that setting ``is_rolling=True`` always produces better quality of imputations
#   but requires a longer training/inference time.

dict_imputers = {
    "tabddpm": ImputerDiffusion(
        model=TabDDPM(num_sampling=5), epochs=10, batch_size=100
    ),
    "tsddpm": ImputerDiffusion(
        model=TsDDPM(num_sampling=5, is_rolling=False),
        epochs=10,
        batch_size=5,
        index_datetime="date",
        freq_str="5D",
    ),
}

comparison = comparator.Comparator(
    dict_imputers,
    selected_columns=df_data.columns,
    generator_holes=missing_patterns.UniformHoleGenerator(n_splits=2),
    metrics=["mae", "kl_columnwise"],
)
results = comparison.compare(df_data)

results.groupby(axis=0, level=0).mean().groupby(axis=0, level=0).mean()

# %%
# [1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. `Denoising diffusion probabilistic models.
# <https://arxiv.org/abs/2006.11239>`_
# Advances in neural information processing systems 33 (2020): 6840-6851.
