# """
# ============================================
# Tutorial for hole generation in tabular data
# ============================================

# In this tutorial, we show how to use the different hole generator classes
# in a time series data case. In particular, we show how to use the
# :class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator`,
# :class:`~qolmat.benchmark.missing_patterns.GeometricHoleGenerator`,
# :class:`~qolmat.benchmark.missing_patterns.EmpiricalHoleGenerator`,
# :class:`~qolmat.benchmark.missing_patterns.MultiMarkovHoleGenerator`
# and :class:`~qolmat.benchmark.missing_patterns.GroupedHoleGenerator`
# classes.
# We use Beijing Multi-Site Air-Quality Data Set.
# It consists in hourly air pollutants data from 12 chinese nationally-controlled
# air-quality monitoring sites.
# """
# from io import BytesIO
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import requests
# import zipfile

# from qolmat.benchmark import missing_patterns
# from qolmat.utils import data

# # %%
# # 1. Data
# # ---------------------------------------------------------------
# # We use the public Beijing Multi-Site Air-Quality Data Set.
# # It consists in hourly air pollutants data from 12 chinese nationally-controlled
# # air-quality monitoring sites.The original data from which the
# # features were extracted comes from
# # https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
# # For the purpose of this notebook,
# # we corrupt the data, with the ``qolmat.utils.data.add_holes`` function.
# # In this way, each column has missing values.

# zip_file_url = (
#     "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
# )
# zip_filename = "PRSA2017_Data_20130301-20170228.zip"

# response = requests.get(zip_file_url)
# if response.status_code == 200:
#     outer_zip_data = BytesIO(response.content)
# else:
#     print("Failed to fetch the zip file. Status code:", response.status_code)
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
#             # Read the CSV file into a DataFrame
#             with inner_zip.open(file_name) as csv_file:
#                 df_data.append(pd.read_csv(csv_file))

#     df_data = pd.concat(df_data, ignore_index=True)


# # %%
# # The dataset contains 18 columns. For simplicity,
# # we only consider some.
# df_data["date"] = pd.to_datetime(df_data[["year", "month", "day", "hour"]])
# df_data = df_data.set_index(["station", "date"])
# columns = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
# df_data = df_data[columns]

# df = data.add_holes(df_data, ratio_masked=0.2, mean_size=120)
# cols_to_impute = df.columns

# # %%
# # Let's visualise the mask (i.e. missing values) of this dataset.
# # Missing values are in white, while observed ones ae in black.

# plt.figure(figsize=(15, 4))
# plt.imshow(df.notna().values.T, aspect="auto", cmap="binary", interpolation="none")
# plt.yticks(range(len(df.columns)), df.columns)
# plt.xlabel("Samples", fontsize=12)
# plt.grid(False)
# plt.show()

# # %%
# # 2. Hole generators
# # ---------------------------------------------------------------
# # Given a pandas dataframe `df`, the aim of a hole generator
# # is to create ``n_splits`` masks, i.e. a list of dataframes,
# # where each dataframe has the same dimension
# # as `df` with missing entries `np.nan`. The missing entries of the mask
# # cannot be missing in the initial dataframe.
# # This is achieved witht the ``split`` function.
# # For each method, we will generate 10 percent missing values, i.e.
# # ``ratio_masked=0.1``, and we will generate missing values
# # for all the columns in the dataframe, i.e. ``subset=df.columns``.
# # Since the exercise here is simply to show how to generate missing data,
# # the ``n_splits`` argument is not important.
# # We therefore set it to 1.
# # Let's just define a funciton to visualise the additional
# # missing values.


# def visualise_missing_values(df_init: pd.DataFrame, df_mask: pd.DataFrame):
#     """Visualise the missing values in the final dataframe
#     with different colors for initial (white) and
#     additional (red) missing values.

#     Parameters
#     ----------
#     df_init : pd.DataFrame
#         initial dataframe
#     df_mask : pd.DataFrame
#         masked dataframe
#     """
#     df_tot = df_init.copy()
#     df_tot[df_init.notna()] = 0
#     df_tot[df_init.isna()] = 2
#     df_mask = np.invert(df_mask).astype("int")
#     df_tot += df_mask
#     colorsList = [(1, 0, 0), (0, 0, 0), (1, 1, 1)]
#     custom_cmap = matplotlib.colors.ListedColormap(colorsList)
#     plt.figure(figsize=(15, 4))
#     plt.imshow(df_tot.values.T, aspect="auto", cmap=custom_cmap, interpolation="none")
#     plt.yticks(range(len(df_tot.columns)), df_tot.columns)
#     plt.xlabel("Samples", fontsize=12)
#     plt.grid(False)
#     plt.show()


# # %%
# # a. Uniform Hole Generator
# # ***************************************************************
# # The holes are generated randomly, using the ``resample`` method of scikit learn.
# # Holels are created column by column. This metohd is implemented in the
# # :class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator` class.
# # Note this class is more suited for tabular datasets.

# generator = missing_patterns.UniformHoleGenerator(
# n_splits=1, subset=df.columns, ratio_masked=0.1
# )
# df_mask = generator.split(df)[0]

# print("Pourcentage of additional missing values:")
# print(round((df_mask.sum() / len(df_mask)) * 100, 2))
# visualise_missing_values(df, df_mask)

# # %%
# # Just to illustrate, imagine we have columns without missing values.
# # In this case, there is no point to create hole in these columns.
# # So all we need to do is pass in the argument the name of the columns
# # for which we want to create gaps, for example,

# generator = missing_patterns.UniformHoleGenerator(
#     n_splits=1,
#     subset=["PRES", "DEWP"],
#     ratio_masked=0.1,
# )
# df_mask = generator.split(df)[0]

# print("Pourcentage of additional missing values:")
# print(round((df_mask.sum() / len(df_mask)) * 100, 2))
# visualise_missing_values(df, df_mask)

# # %%
# # b. Geometric Hole Generator
# # ***************************************************************
# # The holes are generated following a Markov 1D process.
# # Holes are created column by column. The transition matrix of the
# # one-dimensional Markov process is learned from the data.
# # This metohd is implemented in the
# # :class:`~qolmat.benchmark.missing_patterns.UniformHoleGenerator` class.

# generator = missing_patterns.GeometricHoleGenerator(
#     n_splits=1, subset=cols_to_impute, ratio_masked=0.1
# )
# df_mask = generator.split(df)[0]

# print("Pourcentage of additional missing values:")
# print(round((df_mask.sum() / len(df_mask)) * 100, 2))
# visualise_missing_values(df, df_mask)

# # %%
# # c. Empirical Hole Generator
# # ***************************************************************
# # The distribution of holes is learned from the data.
# # The distributions of holes are learned column by column; so you need to fit
# # the generator to the data.
# # This metohd is implemented in the
# # :class:`~qolmat.benchmark.missing_patterns.EmpiricalHoleGenerator` class.
# # We specify ``groups=("station",)`` which means a distribution
# # is learned on each group: here on each station.

# generator_holes = missing_patterns.EmpiricalHoleGenerator(
#     n_splits=1, subset=df.columns, ratio_masked=0.1, groups=("station",)
# )
# df_mask = generator_holes.split(df)[0]

# print("Pourcentage of additional missing values:")
# print(round((df_mask.sum() / len(df_mask)) * 100, 2))
# visualise_missing_values(df, df_mask)

# # %%
# # d. Multi Markov Hole Generator
# # ***************************************************************
# # The holes are generated according to a Markov process.
# # Each line of the dataframe mask (np.nan) represents a state of the Markov chain.
# # Note it is also more difficult to achieve exactly the required
# # missing data ratio.
# # This metohd is implemented in the
# # :class:`~qolmat.benchmark.missing_patterns.MultiMarkovHoleGenerator` class.

# generator_holes = missing_patterns.MultiMarkovHoleGenerator(
#     n_splits=1, subset=df.columns, ratio_masked=0.1
# )
# df_mask = generator_holes.split(df)[0]

# print("Pourcentage of additional missing values:")
# print(round((df_mask.sum() / len(df_mask)) * 100, 2))
# visualise_missing_values(df, df_mask)

# # %%
# # d. Grouped Hole Generator
# # ***************************************************************
# # The holes are generated according to the groups defined by the user.
# # This metohd is implemented in the
# # :class:`~qolmat.benchmark.missing_patterns.GroupedHoleGenerator` class.

# generator_holes = missing_patterns.GroupedHoleGenerator(
#     n_splits=1, subset=df.columns, ratio_masked=0.1, groups=("station",)
# )
# df_mask = generator_holes.split(df)[0]

# print("Pourcentage of additional missing values:")
# print(round((df_mask.sum() / len(df_mask)) * 100, 2))
# visualise_missing_values(df, df_mask)
