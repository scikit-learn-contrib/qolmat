# ######################
# # Evaluation metrics #
# ######################


# def test_mean_squared_error() -> None:
#     df1 = pd.DataFrame(
#         data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
#         columns=["var1", "var2", "var3"],
#     )

#     df2 = pd.DataFrame(
#         data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
#         columns=["var1", "var2", "var3"],
#     )
#     assert_series_equal(
#         utils.mean_squared_error(df1, df2),
#         pd.Series([94, 58, 25], index=["var1", "var2", "var3"]),
#     )


# def test_mean_absolute_error() -> None:
#     df1 = pd.DataFrame(
#         data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
#         columns=["var1", "var2", "var3"],
#     )

#     df2 = pd.DataFrame(
#         data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
#         columns=["var1", "var2", "var3"],
#     )
#     assert utils.mean_absolute_error(df1, df2, columnwise_evaluation=False) == 33


# def test_weighted_mean_absolute_percentage_error() -> None:
#     df1 = pd.DataFrame(
#         data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
#         columns=["var1", "var2", "var3"],
#     )

#     df2 = pd.DataFrame(
#         data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
#         columns=["var1", "var2", "var3"],
#     )

#     assert (
#         round(
#             utils.weighted_mean_absolute_percentage_error(df1, df2, columnwise_evaluation=False),
#             4,
#         )
#         == 0.4484
#     )


# def test_wasser_distance() -> None:
#     df1 = pd.DataFrame(
#         data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
#         columns=["var1", "var2", "var3"],
#     )

#     df2 = pd.DataFrame(
#         data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
#         columns=["var1", "var2", "var3"],
#     )
#     assert_series_equal(
#         utils.wasser_distance(df1, df2),
#         pd.Series([4, 2.5, 1.75], index=["var1", "var2", "var3"]),
#     )


# def test_kl_divergence() -> None:
#     df1 = pd.DataFrame(
#         data=[[1, 2, 3], [6, 4, 2], [7, 8, 9], [10, 10, 12]],
#         columns=["var1", "var2", "var3"],
#     )

#     df2 = pd.DataFrame(
#         data=[[1, 2, 3], [5, 2, 3], [1, 8, 9], [3, 4, 6]],
#         columns=["var1", "var2", "var3"],
#     )
#     assert_series_equal(
#         utils.kl_divergence(df1, df2, columnwise_evaluation=True),
#         pd.Series([17.960112, 17.757379, 17.757379], index=["var1", "var2", "var3"]),
#     )
#     assert round(utils.kl_divergence(df1, df2, columnwise_evaluation=False), 4) == 14.0112


# def test_frechet_distance() -> None:
#     df1 = pd.DataFrame(
#         data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
#         columns=["var1", "var2", "var3"],
#     )

#     df2 = pd.DataFrame(
#         data=[[1, 2, 3], [1, 2, 3], [1, 8, 9], [3, 4, 8]],
#         columns=["var1", "var2", "var3"],
#     )
#     assert round(utils.frechet_distance(df1, df2, normalized=False), 4) == 41.6563
#     assert round(utils.frechet_distance(df1, df2, normalized=True), 4) == 1.9782
