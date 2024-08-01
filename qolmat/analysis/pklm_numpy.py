# Non industrial implementation of the PKLM test
import time
import random
import line_profiler
from itertools import product
from joblib import Parallel, delayed, dump, load
from typing import Tuple, Optional



import pandas as pd
import numpy as np


from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def draw_A_B(matrix: np.ndarray):
    _, p = matrix.shape
    k = np.random.randint(1, p)
    a_indexes = np.random.choice(range(p), size=k, replace=False)
    b_index = np.random.choice(np.setdiff1d(np.arange(p), a_indexes))
    return a_indexes, b_index


def check_A_B(matrix, A, B):
    col_b_na = matrix[~np.isnan(matrix[:,A]).any(axis=1)][:, B]
    is_nan = np.isnan(col_b_na).any()
    is_values = (~np.isnan(col_b_na)).any()
    return is_nan and is_values


def draw_proj(matrix):
    result = False
    while not result:
        A, B = draw_A_B(matrix)
        result = check_A_B(matrix, A, B)
    return A, B


def build_dataset(matrix, A, B):
    X = matrix[~np.isnan(matrix[:,A]).any(axis=1)][:, A]
    y = np.where(np.isnan(matrix[~np.isnan(matrix[:,A]).any(axis=1)][:, B]), 1, 0)
    return X, y


def build_label(matrix, perm, A, B):
    return perm[~np.isnan(matrix[:,A]).any(axis=1), B]


def get_oob_probabilities(X, y, nb_trees_per_proj: int):
    clf = RandomForestClassifier(
        n_estimators=nb_trees_per_proj,
        max_features=None,
        min_samples_split=10,
        bootstrap=True,
        oob_score=True,
    )
    clf.fit(X, y)
    return clf.oob_decision_function_


def get_scores(X, y):
    clf = HistGradientBoostingClassifier()
    clf.fit(X, y)
    return clf.predict_proba(X)


def U_hat(oob_probabilities: np.ndarray, labels: np.ndarray):
    oob_probabilities = np.clip(oob_probabilities, 1e-9, 1-1e-9)

    unique_labels = np.unique(labels)
    label_matrix = (labels[:, None] == unique_labels).astype(int)
    p_true = oob_probabilities * label_matrix
    p_false = oob_probabilities * (1 - label_matrix)

    # for g=0
    p0_0 = p_true[:, 0][np.where(p_true[:, 0] != 0.)]
    p0_1 = p_false[:, 0][np.where(p_false[:, 0] != 0.)]
    
    # for g=1
    p1_1 = p_true[:, 1][np.where(p_true[:, 1] != 0.)]
    p1_0 = p_false[:, 1][np.where(p_false[:, 1] != 0.)]

    if unique_labels.shape[0] == 1:
        if unique_labels[0] == 0:
            n0 = labels.shape[0]
            return np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p1_0 / (1 - p1_0)).sum() / n0

        else:
            n1 = labels.shape[0]
            return np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p0_1 / (1 - p0_1)).sum() / n1
    
    n0, n1 = label_matrix.sum(axis=0)
    u_0 = np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p0_1 / (1 - p0_1)).sum() / n1
    u_1 = np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p1_0 / (1 - p1_0)).sum() / n0
    
    return u_0 + u_1


def process_perm(matrix, M_perm, A, B, oob_probs):
    y = build_label(matrix, M_perm, A, B)
    return U_hat(oob_probs, y)


def process_all(matrix, list_perm, A, B, nb_trees_per_proj):
    X, y = build_dataset(matrix, A, B)
    #oob_prob = get_oob_probabilities(X, y, nb_trees_per_proj)
    oob_prob = get_scores(X, y)
    u_hat = U_hat(oob_prob, y)
    result_u_sigmas = Parallel(n_jobs=-1)(delayed(process_perm)(
        matrix,
        M_perm,
        A,
        B,
        oob_prob
    ) for M_perm in list_perm)
    return u_hat, result_u_sigmas

@line_profiler.profile
def PKLMtest(
    matrix,
    nb_projections: int = 100,
    nb_permutations: int = 30,
    nb_trees_per_proj: int = 200,
) -> float:
    M = np.isnan(matrix).astype(int)

    list_proj = [draw_proj(matrix) for _ in range(nb_projections)]
    list_perm = [np.random.permutation(M) for _ in range(nb_permutations)]

    U = 0.
    list_U_sigma = [0. for _ in range(nb_permutations)]

    parallel_results = Parallel(n_jobs=-1)(delayed(process_all)(
        matrix,
        list_perm,
        A,
        B,
        nb_trees_per_proj
    ) for A, B in list_proj)

    for U_projection, results in parallel_results:
        U += U_projection
        list_U_sigma = [x + y for x, y in zip(list_U_sigma, results)]

    U = U / nb_projections
    list_U_sigma = [x / nb_permutations for x in list_U_sigma]

    p_value = 1
    for u_sigma in list_U_sigma:
        if u_sigma >= U:
            p_value += 1
    return p_value / (nb_permutations + 1)


if __name__ == "__main__":
    df = pd.read_csv("qolmat/analysis/df7.csv").to_numpy()
    start_time = time.time()
    p_v = PKLMtest(df)
    print(p_v)
    print("--- %s seconds ---" % (time.time() - start_time))
