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



def draw_A_B(matrix):
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
    A = [0]
    B = [0]
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
    return np.where(np.isnan(perm[~np.isnan(matrix[:,A]).any(axis=1)][:, B]), 1, 0)


def get_oob_probabilities(X, y, nb_trees_per_proj: int, proj_index: int):
    clf = RandomForestClassifier(
        n_estimators=nb_trees_per_proj,
        max_features=None,
        min_samples_split=10,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    clf.fit(X, y)
    # TODO : Créer les fichiers
    dump(clf.oob_decision_function_, f'examples/run/optim/oob_{proj_index}.joblib')
    return clf.oob_decision_function_


def U_hat(oob_probabilities: np.ndarray, labels: pd.Series):
    oob_probabilities = np.clip(oob_probabilities, 1e-9, 1-1e-9)
    unique_labels = np.unique(labels)
    label_matrix = (labels[:, None] == unique_labels).astype(int)
    p = np.sum(oob_probabilities * label_matrix, axis=0) / label_matrix.sum(axis=0)
    p_a = np.sum(oob_probabilities * (1 - label_matrix), axis=0) / (1 - label_matrix).sum(axis=0)
    return np.mean(np.log(p / (1 - p)) - np.log(p_a / (1 - p_a)))


def process(A, B, matrix, nb_trees_per_proj, proj_index):
    X, y = build_dataset(matrix, A, B)
    oob_prob = get_oob_probabilities(X, y, nb_trees_per_proj, proj_index)
    return U_hat(oob_prob, y)

@line_profiler.profile
def PKLMtest(
    matrix,
    nb_projections: int = 100,
    nb_permutations: int = 30,
    nb_trees_per_proj: int = 200,
) -> float:
    list_proj = [draw_proj(matrix) for _ in range(nb_projections)]
    results = Parallel(n_jobs=-1)(delayed(process)(
        A,
        B,
        matrix,
        nb_trees_per_proj,
        proj_index
    ) for proj_index, (A, B) in enumerate(list_proj))
    U = np.mean(results)

    list_perm = [np.random.permutation(matrix) for _ in range(nb_permutations)]
    dict_u_sigma = {k: 0 for k in range(nb_permutations)}

    for index, ((A, B), perm) in enumerate(product(list_proj, list_perm)):
        y = build_label(matrix, perm, A, B)
        index_proj = index//nb_permutations
        index_perm = index%nb_permutations
        oob_probs = load(f'examples/run/optim/oob_{index_proj}.joblib')
        dict_u_sigma[index_perm] += U_hat(oob_probs, y)

    # TODO : Penser à écraser les oobs prob

    list_U_sigma = [x / nb_permutations for _, x in dict_u_sigma.items()]

    p_value = 1
    for u_sigma in list_U_sigma:
        if u_sigma >= U:
            p_value += 1
    return p_value / (nb_permutations + 1)


if __name__ == "__main__":
    df = pd.read_csv("qolmat/analysis/df2.csv").to_numpy()
    start_time = time.time()
    p_v = PKLMtest(df)
    print(p_v)
    print("--- %s seconds ---" % (time.time() - start_time))
