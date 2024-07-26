# Non industrial implementation of the PKLM test
import random
import line_profiler
from joblib import Parallel, delayed
from typing import Tuple, Optional



import pandas as pd
import numpy as np


from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

@line_profiler.profile
def draw_A_B(matrix):
    _, p = matrix.shape
    k = np.random.randint(1, p)
    a_indexes = np.random.choice(range(p), size=k, replace=False)
    b_index = np.random.choice(np.setdiff1d(np.arange(p), a_indexes))
    return a_indexes, b_index

@line_profiler.profile
def check_A_B(matrix, A, B):
    col_b_na = matrix[~np.isnan(matrix[:,A]).any(axis=1)][:, B]
    is_nan = np.isnan(col_b_na).any()
    is_values = (~np.isnan(col_b_na)).any()
    return is_nan and is_values

@line_profiler.profile
def draw_proj(matrix):
    A = [0]
    B = [0]
    result = False
    while not result:
        A, B = draw_A_B(matrix)
        result = check_A_B(matrix, A, B)
    return A, B

@line_profiler.profile
def build_dataset(matrix, A, B):
    X = matrix[~np.isnan(matrix[:,A]).any(axis=1)][:, A]
    y = np.where(np.isnan(matrix[~np.isnan(matrix[:,A]).any(axis=1)][:, B]), 1, 0)
    return X, y

@line_profiler.profile
def get_oob_probabilities(X, y, nb_trees_per_proj: int):
    clf = RandomForestClassifier(
        n_estimators=nb_trees_per_proj,
        max_features=None,
        min_samples_split=10,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    clf.fit(X, y)

    return clf.oob_decision_function_

@line_profiler.profile
def U_hat(oob_probabilities: np.ndarray, labels: pd.Series):
    oob_probabilities = np.clip(oob_probabilities, 1e-9, 1-1e-9)
    unique_labels = np.unique(labels)
    label_matrix = (labels[:, None] == unique_labels).astype(int)
    p = np.sum(oob_probabilities * label_matrix, axis=0) / label_matrix.sum(axis=0)
    p_a = np.sum(oob_probabilities * (1 - label_matrix), axis=0) / (1 - label_matrix).sum(axis=0)
    return np.mean(np.log(p / (1 - p)) - np.log(p_a / (1 - p_a)))

@line_profiler.profile
def PKLMtest(
    matrix,
    nb_projections: int = 100,
    nb_permutations: int = 30,
    nb_trees_per_proj: int = 200,
) -> float:
    list_proj = [draw_proj(matrix) for _ in range(nb_projections)]
    U = 0.0
    for A, B in list_proj:
        X, y = build_dataset(matrix, A, B)
        oob_prob = get_oob_probabilities(X, y, nb_trees_per_proj)
        U += U_hat(oob_prob, y)
    return U


if __name__ == "__main__":
    df = pd.read_csv("qolmat/analysis/df6.csv").to_numpy()
    PKLMtest(df)
