# """
# =========================
# Benchmark for time series
# =========================

# # In this tutorial, we show how to use Qolmat to benchmark several
# # imputation methods and a multivariate time series dataset.
# # We use Beijing Multi-Site Air-Quality Data Set.
# # It consists in hourly air pollutants data from 12 chinese nationally-controlled
# # air-quality monitoring sites.
# """

# # %%
# # First import some libraries

# import hyperopt as ho
# import numpy as np
# import pandas as pd
# import scipy
# from hyperopt.pyll.base import Apply as hoApply

# np.random.seed(1234)
# import matplotlib.ticker as plticker
# from matplotlib import pyplot as plt

# tab10 = plt.get_cmap("tab10")

# import zipfile
# from io import BytesIO

# import requests
# from qolmat.benchmark import comparator, hyperparameters, missing_patterns
# from qolmat.benchmark.metrics import kl_divergence
# from qolmat.imputations import imputers
# from qolmat.utils import data, plot, utils
# from sklearn.ensemble import (
#     ExtraTreesRegressor,
#     HistGradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression

# # %%
# # 1. Data
# # ---------------------------------------------------------------
# # We use the public Beijing Multi-Site Air-Quality Data Set.
# # It consists in hourly air pollutants data from 12 chinese nationally-controlled
# # air-quality monitoring sites.The original data from which the
# # features were extracted comes from
# # https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
# # In this way, each column has missing values.
# # We group the data by day and only consider 5 columns.
# # For the purpose of this notebook,
# # we corrupt the data, with the ``qolmat.utils.data.add_holes`` function.


# zip_file_url = (
#     "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
# )
# zip_filename = "PRSA2017_Data_20130301-20170228.zip"

# response = requests.get(zip_file_url)
# if response.status_code == 200:
#     outer_zip_data = BytesIO(response.content)
# else:
#     print("Failed to fetch the outer zip file. Status code:", response.status_code)
#     exit()

# outer_zip = zipfile.ZipFile(outer_zip_data, "r")
# if zip_filename in outer_zip.namelist():
#     inner_zip_data = BytesIO(outer_zip.read(zip_filename))
#     outer_zip.close()
# else:
#     print(f"The inner zip file '{zip_filename}' was not found in the outer zip file.")
#     outer_zip.close()
#     exit()

# with zipfile.ZipFile(inner_zip_data, "r") as inner_zip:
#     df_data = []
#     for file_name in inner_zip.namelist():
#         if file_name.endswith(".csv"):
#             with inner_zip.open(file_name) as csv_file:
#                 df_data.append(pd.read_csv(csv_file))

#     df_data = pd.concat(df_data, ignore_index=True)

# df_data["date"] = pd.to_datetime(df_data[["year", "month", "day"]])
# df_data = df_data.set_index(["station", "date"])
# df_data = df_data[["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]]
# df_data = df_data.groupby(level=["station", "date"]).mean()
# cols_to_impute = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
# df = data.add_holes(df_data, ratio_masked=0.15, mean_size=50)
# # %%
# # Let's take a look a one station, for instance "Aotizhongxin"

# station = "Aotizhongxin"
# fig, ax = plt.subplots(5, 1, figsize=(13, 8))
# for i, col in enumerate(cols_to_impute):
#     ax[i].plot(df.loc[station, col])
#     ax[i].set_ylabel(col)
# fig.align_labels()
# ax[0].set_title(station, fontsize=14)
# plt.tight_layout()
# plt.show()

# # %%
# # 2. Imputation methods
# # ---------------------------------------------------------------
# #  All presented methods are group-wise: here each station is imputed independently.
# # For example ImputerMean computes the mean of each variable in each station and uses
# # the result for imputation; ImputerInterpolation interpolates termporal
# # signals corresponding to each variable on each station.

# ratio_masked = 0.1

# imputer_median = imputers.ImputerMedian(groups=("station",))
# imputer_locf = imputers.ImputerLOCF(groups=("station",))
# imputer_interpol = imputers.ImputerInterpolation(groups=("station",), method="linear")
# imputer_shuffle = imputers.ImputerShuffle(groups=("station",))
# imputer_residuals = imputers.ImputerResiduals(
#     groups=("station",),
#     period=365,
#     model_tsa="additive",
#     extrapolate_trend="freq",
#     method_interpolation="linear",
# )
# imputer_rpca = imputers.ImputerRPCA(
#     groups=("station",), columnwise=False, max_iterations=500, tau=2, lam=0.05
# )
# imputer_ou = imputers.ImputerEM(
#     groups=("station",),
#     model="multinormal",
#     method="sample",
#     max_iter_em=30,
#     n_iter_ou=15,
#     dt=1e-3,
# )
# imputer_tsou = imputers.ImputerEM(
#     groups=("station",),
#     model="VAR",
#     method="sample",
#     max_iter_em=30,
#     n_iter_ou=15,
#     dt=1e-3,
#     p=1,
# )
# imputer_tsmle = imputers.ImputerEM(
#     groups=("station",),
#     model="VAR",
#     method="mle",
#     max_iter_em=30,
#     n_iter_ou=15,
#     dt=1e-3,
#     p=1,
# )
# imputer_mice = imputers.ImputerMICE(
#     groups=("station",),
#     estimator=LinearRegression(),
#     sample_posterior=False,
#     max_iter=100,
# )

# generator_holes = missing_patterns.EmpiricalHoleGenerator(
#     n_splits=4, groups=("station",), subset=cols_to_impute, ratio_masked=ratio_masked
# )

# dict_imputers = {
#     "median": imputer_median,
#     "interpolation": imputer_interpol,
#     "shuffle": imputer_shuffle,
#     "residuals": imputer_residuals,
#     "OU": imputer_ou,
#     "TSOU": imputer_tsou,
#     "TSMLE": imputer_tsmle,
#     # "RPCA": imputer_rpca,
#     "locf": imputer_locf,
#     "mice_ols": imputer_mice,
# }
# n_imputers = len(dict_imputers)

# comparison = comparator.Comparator(
#     dict_imputers,
#     cols_to_impute,
#     generator_holes=generator_holes,
#     metrics=["mae", "wmape", "KL_columnwise", "ks_test", "dist_corr_pattern"],
#     max_evals=10,
# )
# results = comparison.compare(df)
# results.style.highlight_min(color="lime", axis=1)
