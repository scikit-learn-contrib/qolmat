# Non industrial implementation of the PKLM test
import random
from typing import Tuple


import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier


### Draw projections ###
# - the 'size_resp_set' param is set to to 2 ah this moment.


def draw_A(df: pd.DataFrame) -> pd.DataFrame:
    _, n_col = df.shape
    if n_col == 2:
        return df.sample(n=1, axis=1)
    return df.sample(np.random.randint(1, n_col - 1), axis=1)


def draw_B(M: pd.DataFrame) -> str:
    S = M.nunique()
    S = S[S > 1]
    list_col = S.index.tolist()
    return random.choice(list_col)


def check_binary_class(M: pd.DataFrame) -> bool:
    S = M.nunique()
    return S.sum() > S.shape[0]


def build_M_proj(df_A: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    indexes = df_A.dropna().index
    return M.loc[indexes, :].drop(columns=df_A.columns.tolist())


def draw_A_B(df: pd.DataFrame) -> Tuple[list, list]:
    M = 1 * df.isnull()
    temp_res = False
    while not temp_res:
        df_A = draw_A(df)
        M_proj = build_M_proj(df_A, M)
        temp_res = check_binary_class(M_proj)
    col_B = draw_B(M_proj)
    return df_A.columns.tolist(), [col_B]


def check_B(df: pd.DataFrame, M: pd.DataFrame, col_features: list, col_name: str) -> bool:
    M_proj = build_M_proj(df[col_features], M)
    return pd.unique(M_proj[col_name]).shape[0] > 1


def create_binary_classif_df(
    df: pd.DataFrame,
    M: pd.DataFrame,
    col_feature: list,
    col_target: list,
) -> pd.DataFrame:
    df = df[col_feature].dropna()
    M_proj = build_M_proj(df, M)[col_target]
    M_proj = M_proj.rename(columns={col_target[0]: "target"})
    return pd.concat([df, M_proj], axis=1).reset_index(drop=True)


def get_oob_probabilities(df: pd.DataFrame, nb_trees_per_proj: int):
    X, y = df.loc[:, df.columns != "target"], df["target"]
    clf = RandomForestClassifier(
        n_estimators=nb_trees_per_proj,
        max_features=None,
        min_samples_split=10,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    clf.fit(X, y)

    return clf.oob_decision_function_


def U_hat(oob_probabilities, labels):

    def trunc_proba(p: float) -> float:
        return min(max(p, 1e-9), 1-1e-9)

    v_trunc_proba = np.vectorize(trunc_proba)
    oob_probabilities = v_trunc_proba(oob_probabilities)

    U = 0

    for ind in range(len(labels.unique())):
        elements = labels[labels == ind].index
        p = oob_probabilities[elements, ind]
        
        elements_a = labels[labels != ind].index
        p_a = oob_probabilities[elements_a, ind]
        U += np.mean(np.log(p/(1 - p))) - np.mean(np.log(p_a/(1 - p_a)))

    return U


def PKLMtest(
    df: pd.DataFrame,
    nb_projections: int = 100,
    nb_permutations: int = 30,
    nb_trees_per_proj: int = 200
) -> float:
    M = 1 * df.isnull()
    list_perm_M = [M.sample(frac=1, axis=0).reset_index(drop=True) for _ in range(nb_permutations)]
    dict_res_sigma = {i: 0.0 for i in range(nb_permutations)}
    dict_count_sigma = {i: 0.0 for i in range(nb_permutations)}
    U = 0.0
    for _ in range(nb_projections):
        cols_feature, col_target = draw_A_B(df)
        df_for_classification = create_binary_classif_df(df, M, cols_feature, col_target)
        oob_probabilities = get_oob_probabilities(df_for_classification, nb_trees_per_proj)
        res = U_hat(oob_probabilities, df_for_classification.target)
        U += res
        for idx, M_perm in enumerate(list_perm_M):
            if check_B(df, M_perm, cols_feature, col_target[0]):
                indexes = df[cols_feature].dropna().index
                dict_res_sigma[idx] += U_hat(oob_probabilities, M_perm.loc[indexes, col_target[0]].reset_index(drop=True))
                dict_count_sigma[idx] += 1
            else:
                print("aie")

    U = U / nb_projections
    dict_res_sigma = {k: v / dict_count_sigma[k] for k, v in dict_res_sigma.items()}

    p_value = 1
    for _, u_sigma in dict_res_sigma.items():
        if u_sigma >= U:
            p_value += 1
    return p_value / (nb_permutations + 1)
