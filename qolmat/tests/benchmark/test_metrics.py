import pandas as pd
import numpy as np
import scipy
import dcor

from qolmat.benchmark import metrics

import time

df1 = pd.DataFrame([[1, 2, 3], [4, 5, 7], [10, 14, 22]], columns=["a", "b", "c"])
df2 = df1 + 1


def test_kolmogorov_smirnov_test():
    assert metrics.kolmogorov_smirnov_test(df1, df1).equals(
        pd.Series([0.0, 0.0, 0.0], index=["a", "b", "c"])
    )
    assert metrics.kolmogorov_smirnov_test(df1, df2).equals(
        pd.Series([1 / 3, 1 / 3, 1 / 3], index=["a", "b", "c"])
    )


def test_sum_energy_distances():
    start_time = time.time()
    sum_distances_df1 = np.sum(scipy.spatial.distance.cdist(df1, df1, metric="cityblock"))
    sum_distances_df2 = np.sum(scipy.spatial.distance.cdist(df2, df2, metric="cityblock"))
    sum_distances_df1_df2 = np.sum(scipy.spatial.distance.cdist(df1, df2, metric="cityblock"))
    energy_distance_scipy = 2 * sum_distances_df1_df2 - sum_distances_df1 - sum_distances_df2
    print("Scipy: %s seconds" % (time.time() - start_time))

    start_time = time.time()
    energy_distance_qolmat = metrics.sum_energy_distances(df1, df2)
    print("Qolmat: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    # sum_distances_df1 = np.sum(scipy.spatial.distance.cdist(df1, df1, metric='cityblock'))
    # sum_distances_df2 = np.sum(scipy.spatial.distance.cdist(df2, df2, metric='cityblock'))
    # sum_distances_df1_df2 = np.sum(scipy.spatial.distance.cdist(df1, df2, metric='cityblock'))
    # energy_distance_scipy = 2 * sum_distances_df1_df2 - sum_distances_df1 / 2 - sum_distances_df2 / 2
    # print("Dcor: %s seconds" % (time.time() - start_time))

    assert energy_distance_scipy == energy_distance_qolmat


test_sum_energy_distances()
