# Non industrial implementation of the PKLM test
import random
from typing import Tuple


import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier


def draw_A(df: pd.DataFrame) -> pd.DataFrame:
    _, n_col = df.shape
    if n_col == 2:
        return df.sample(n=1, axis=1)
    return df.sample(np.random.randint(1, n_col - 1), axis=1)


def build_M_proj(df_A: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    indexes = df_A.dropna().index
    return M.loc[indexes, :].drop(columns=df_A.columns.tolist())


def check_binary_class(M: pd.DataFrame) -> bool:
    S = M.nunique()
    return S.sum() > S.shape[0]


def draw_B(M: pd.DataFrame) -> str:
    S = M.nunique()
    S = S[S > 1]
    list_col = S.index.tolist()
    return random.choice(list_col)


def draw_A_B(df: pd.DataFrame, M: pd.DataFrame) -> Tuple[list, list]:
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


def get_oob_probabilities(df: pd.DataFrame):
    X, y = df.loc[:, df.columns != "target"], df["target"]
    clf = RandomForestClassifier(
        n_estimators=200,
        max_features=None,
        min_samples_split=10,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    clf.fit(X, y)
    return clf.oob_decision_function_


def U_hat(df: pd.DataFrame, M: pd.DataFrame, A: list, B: list) -> float:
    df_class = create_binary_classif_df(df, M, A, B)
    oob_probs = get_oob_probabilities(df_class)

    def my_func(x: float, eps: float = 1e-5) -> float:
        return np.log((x + eps) / (1 - x + eps))

    vfunc = np.vectorize(my_func)

    element_0 = df_class[df_class["target"] == 0].index
    element_1 = df_class[df_class["target"] == 1].index

    return 1 / element_1.shape[0] * np.sum(vfunc(oob_probs[element_1, 1])) - 1 / element_0.shape[
        0
    ] * np.sum(vfunc(oob_probs[element_0, 0]))


def PKLMtest(
    df: pd.DataFrame,
    nb_projections: int,
    nb_permutations: int,
) -> float:
    M = 1 * df.isnull()
    list_perm_M = [M.sample(frac=1, axis=0).reset_index(drop=True) for _ in range(nb_permutations)]
    dict_res_sigma = {i: 0.0 for i in range(nb_permutations)}
    dict_count_sigma = {i: 0.0 for i in range(nb_permutations)}
    U = 0.0
    for _ in range(nb_projections):
        cols_feature, col_target = draw_A_B(df, M)
        res = U_hat(df, M, cols_feature, col_target)
        U += res
        for idx, M_perm in enumerate(list_perm_M):
            if check_B(df, M_perm, cols_feature, col_target[0]):
                dict_res_sigma[idx] += U_hat(df, M_perm, cols_feature, col_target)
                dict_count_sigma[idx] += 1

    U = U / nb_projections
    dict_res_sigma = {k: v / dict_count_sigma[k] for k, v in dict_res_sigma.items()}

    p_value = 1
    for ind_perm, u_sigma in dict_res_sigma.items():
        if u_sigma >= U:
            p_value += 1
    return p_value / (nb_permutations + 1)
