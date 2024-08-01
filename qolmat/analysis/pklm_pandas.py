# Non industrial implementation of the PKLM test
import time
import random
from joblib import Parallel, delayed
from typing import Tuple, Optional



import pandas as pd
import numpy as np


from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


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
        random_state=42
    )
    clf.fit(X, y)

    return clf.oob_decision_function_


def get_scores(df: pd.DataFrame):
    X, y = df.loc[:, df.columns != "target"], df["target"]
    clf = HistGradientBoostingClassifier()
    clf.fit(X, y)

    return clf.predict_proba(X)


def U_hat(oob_probabilities, labels):
    oob_probabilities = np.clip(oob_probabilities, 1e-9, 1-1e-9)
    
    unique_labels = labels.unique()
    label_matrix = (labels.values[:, None] == unique_labels).astype(int)
    p_true = oob_probabilities * label_matrix
    p_false = oob_probabilities * (1 - label_matrix)
    
    # for g=0
    p0_0 = p_true[:, 0][np.where(p_true[:, 0] != 0.)]
    p0_1 = p_false[:, 0][np.where(p_false[:, 0] != 0.)]
    
    # for g=1
    p1_1 = p_true[:, 1][np.where(p_true[:, 1] != 0.)]
    p1_0 = p_false[:, 1][np.where(p_false[:, 1] != 0.)]
    
    if len(labels.unique()) == 1:
        if labels.unique() == 0:
            n0 = labels.shape[0]
            return np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p1_0 / (1 - p1_0)).sum() / n0

        elif labels.unique() == 1:
            n1 = labels.shape[0]
            return np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p0_1 / (1 - p0_1)).sum() / n1
    
    n0, n1 = label_matrix.sum(axis=0)
    u_0 = np.log(p0_0 / (1 - p0_0)).sum() / n0 - np.log(p0_1 / (1 - p0_1)).sum() / n1
    u_1 = np.log(p1_1 / (1 - p1_1)).sum() / n1 - np.log(p1_0 / (1 - p1_0)).sum() / n0
    
    return u_0 + u_1


def process_perm(M_perm, df, cols_feature, col_target, oob_probs):
    indexes = df[cols_feature].dropna().index
    return U_hat(oob_probs, M_perm.loc[indexes, col_target[0]].reset_index(drop=True))



def process_all(df, M, list_perm_M, nb_trees_per_proj):
    cols_feature, col_target = draw_A_B(df)
    df_for_classification = create_binary_classif_df(df, M, cols_feature, col_target)
    # Here to switch the classifier
    oob_probabilities = get_oob_probabilities(df_for_classification, nb_trees_per_proj)
    # oob_probabilities = get_scores(df_for_classification)
    u_hat = U_hat(oob_probabilities, df_for_classification.target)
    results = Parallel(n_jobs=-1)(delayed(process_perm)(
        M_perm,
        df,
        cols_feature,
        col_target,
        oob_probabilities
    ) for M_perm in list_perm_M)
    return u_hat, results



def PKLMtest(
    df: pd.DataFrame,
    nb_projections: int = 100,
    nb_permutations: int = 30,
    nb_trees_per_proj: int = 200,
) -> float:
    M = 1 * df.isnull()
    list_perm_M = [M.sample(frac=1, axis=0).reset_index(drop=True) for _ in range(nb_permutations)]
    list_U_sigma = [0 for _ in range(nb_permutations)]
    U = 0.0
    parallel_results = Parallel(n_jobs=-1)(delayed(process_all)(
        df,
        M,
        list_perm_M,
        nb_trees_per_proj
    ) for _ in range(nb_projections))

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
    df = pd.read_csv("qolmat/analysis/df_for_bug.csv")
    start_time = time.time()
    p_v = PKLMtest(df=df, nb_projections=100, nb_permutations=200)
    print(p_v)
    print("--- %s seconds ---" % (time.time() - start_time))
