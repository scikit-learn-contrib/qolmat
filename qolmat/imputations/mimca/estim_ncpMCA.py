"""Estimate the optimal number of dimensions for MCA using CV or LOO."""


import numpy as np
import pandas as pd
from tqdm import tqdm


def moy_p(V, weights):
    """Compute the weighted mean of a vector, ignoring NaNs.

    Parameters
    ----------
    V : array-like
        Input vector with possible NaN values.
    weights : array-like
        Weights corresponding to each element in V.

    Returns
    -------
    float
        Weighted mean of non-NaN elements.

    """
    mask = ~np.isnan(V)
    total_weight = np.sum(weights[mask])
    if total_weight == 0:
        return 0.0
    return np.sum(V[mask] * weights[mask]) / total_weight

def tab_disjonctif_NA(df):
    """Create a disjunctive table for categorical variables, preserving NaNs.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with categorical and numeric variables.

    Returns
    -------
    DataFrame
        Disjunctive table with one-hot encoding, preserving NaNs.

    """
    df_encoded_list = []
    for col in df.columns:
        if df[col].dtype.name == "category" or df[col].dtype == object:
            df[col] = df[col].astype("category")
            encoded = pd.get_dummies(
                df[col],
                prefix=col,
                prefix_sep="_",
                dummy_na=False,
                dtype=float,
            )
            categories = df[col].cat.categories.tolist()
            col_names = [f"{col}_{cat}" for cat in categories]
            encoded = encoded.reindex(columns=col_names, fill_value=0.0)
            encoded[df[col].isna()] = np.nan
            df_encoded_list.append(encoded)
        else:
            df_encoded_list.append(df[[col]])
    df_encoded = pd.concat(df_encoded_list, axis=1)
    return df_encoded

def prodna(data, noNA, rng):
    """Introduce random missing values into a DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input data.
    noNA : float
        Proportion of missing values to introduce.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    DataFrame
        DataFrame with introduced missing values.

    """
    data = data.copy()
    n_rows, n_cols = data.shape
    total_values = n_rows * n_cols
    n_missing = int(np.floor(total_values * noNA))
    missing_indices = rng.choice(total_values, n_missing, replace=False)
    row_indices = missing_indices // n_cols
    col_indices = missing_indices % n_cols
    for i in range(n_missing):
        row = row_indices[i]
        col = col_indices[i]
        data.iloc[row, col] = np.nan
    return data

def find_category(df_original, tab_disj):
    """Reconstruct original categorical variables from disjunctive table.

    Parameters
    ----------
    df_original : DataFrame
        Original DataFrame with categorical variables.
    tab_disj : DataFrame
        Disjunctive table after imputation.

    Returns
    -------
    DataFrame
        Reconstructed DataFrame with imputed categorical variables.

    """
    df_reconstructed = df_original.copy()
    start_idx = 0
    for col in df_original.columns:
        if df_original[col].dtype.name == "category" or df_original[col].dtype == object: # noqa: E501
            categories = df_original[col].cat.categories.tolist()
            num_categories = len(categories)
            sub_tab = tab_disj.iloc[:, start_idx : start_idx + num_categories]
            max_indices = sub_tab.values.argmax(axis=1)
            df_reconstructed[col] = [categories[idx] for idx in max_indices]
            df_reconstructed[col].replace("__MISSING__", np.nan, inplace=True)
            start_idx += num_categories
        else:
            start_idx += 1
    return df_reconstructed

def imputeMCA(
    don,
    ncp=2,
    method="Regularized",
    row_w=None,
    coeff_ridge=1,
    threshold=1e-6,
    seed=None,
    maxiter=1000,
):
    """Impute missing values in a dataset using MCA.

    Parameters
    ----------
    don : DataFrame
        Input dataset with missing values.
    ncp : int, optional
        Number of principal components for MCA. Default is 2.
    method : str, optional
        Imputation method ('Regularized' or 'EM'). Default is 'Regularized'.
    row_w : array-like, optional
        Row weights. If None, uniform weights are applied. Default is None.
    coeff_ridge : float, optional
        Regularization coefficient for 'Regularized' MCA. Default is 1.
    threshold : float, optional
        Convergence threshold. Default is 1e-6.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    maxiter : int, optional
        Maximum number of iterations for the imputation process.

    Returns
    -------
    dict
        Dictionary containing:
            - "tab_disj": Disjunctive coded table after imputation.
            - "completeObs": Complete dataset with missing values imputed.

    """
    don = pd.DataFrame(don)
    don = don.copy()
    for col in don.columns:
        if not pd.api.types.is_numeric_dtype(don[col]) or don[col].dtype == "bool":  # noqa: E501
            don[col] = don[col].astype("category")
            new_categories = don[col].cat.categories.astype(str)
            don[col] = don[col].cat.rename_categories(new_categories) # noqa: E501
        else:
            unique_values = don[col].dropna().unique()
            if set(unique_values).issubset({0, 1}):
                don[col] = don[col].astype("category")
                new_categories = don[col].cat.categories.astype(str)
                don[col] = don[col].cat.rename_categories(new_categories) # noqa: E501
    if row_w is None:
        row_w = np.ones(len(don)) / len(don)
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()
    tab_disj_NA = tab_disjonctif_NA(don)
    if ncp == 0:
        tab_disj_comp_mean = tab_disj_NA.apply(lambda col: moy_p(col.values, row_w))  # noqa: E501
        tab_disj_comp = tab_disj_NA.fillna(tab_disj_comp_mean)
        completeObs = find_category(don, tab_disj_comp)
        return {"tab_disj": tab_disj_comp, "completeObs": completeObs}
    tab_disj_comp = tab_disj_NA.copy()
    hidden = tab_disj_NA.isna()
    tab_disj_comp.fillna(tab_disj_comp.mean(), inplace=True)
    tab_disj_rec_old = tab_disj_comp.copy()
    nbiter = 0
    continue_flag = True
    while continue_flag:
        nbiter += 1
        M = tab_disj_comp.apply(lambda col: moy_p(col.values, row_w)) / don.shape[1]  # noqa: E501
        M = M.replace({0: np.finfo(float).eps})
        M = M.fillna(np.finfo(float).eps)
        tab_disj_comp_mean = tab_disj_comp.apply(lambda col: moy_p(col.values, row_w))  # noqa: E501
        tab_disj_comp_mean = tab_disj_comp_mean.replace({0: np.finfo(float).eps})  # noqa: E501
        Z = tab_disj_comp.div(tab_disj_comp_mean, axis=1)
        Z_mean = Z.apply(lambda col: moy_p(col.values, row_w))
        Z = Z.subtract(Z_mean, axis=1)
        Zscale = Z.multiply(np.sqrt(M), axis=1)
        U, s, Vt = np.linalg.svd(Zscale.values, full_matrices=False)
        V = Vt.T
        U = U[:, :ncp]
        V = V[:, :ncp]
        s = s[:ncp]
        if method.lower() == "em":
            moyeig = 0
        else:
            if len(s) > ncp:
                moyeig = np.mean(s[ncp:] ** 2)
                moyeig = min(moyeig * coeff_ridge, s[ncp - 1] ** 2)
            else:
                moyeig = 0
        eig_shrunk = (s ** 2 - moyeig) / s
        eig_shrunk = np.maximum(eig_shrunk, 0)
        rec = U @ np.diag(eig_shrunk) @ V.T
        tab_disj_rec = pd.DataFrame(
            rec, columns=tab_disj_comp.columns, index=tab_disj_comp.index
        )
        tab_disj_rec = tab_disj_rec.div(np.sqrt(M), axis=1) + 1
        tab_disj_rec = tab_disj_rec.multiply(tab_disj_comp_mean, axis=1)
        diff = tab_disj_rec - tab_disj_rec_old
        diff_values = diff.values
        hidden_values = hidden.values
        diff_values[~hidden_values] = 0
        relch = np.sum((diff_values**2) * row_w[:, None])
        tab_disj_rec_old = tab_disj_rec.copy()
        tab_disj_comp.values[hidden_values] = tab_disj_rec.values[hidden_values] # noqa: E501
        continue_flag = (relch > threshold) and (nbiter < maxiter)
    completeObs = find_category(don, tab_disj_comp)
    return {"tab_disj": tab_disj_comp, "completeObs": completeObs}

def estim_ncpMCA(
    don,
    ncp_min=0,
    ncp_max=5,
    method="Regularized",
    method_cv="Kfold",
    nbsim=100,
    pNA=0.05,
    ind_sup=None,
    quanti_sup=None,
    quali_sup=None,
    threshold=1e-4,
    verbose=True,
    seed=None
):
    """Estimate the optimal number of dimensions for MCA using CV.

    Parameters
    ----------
    don : DataFrame
        Input data.
    ncp_min : int, optional
        Minimum number of components to test. Default is 0.
    ncp_max : int, optional
        Maximum number of components to test. Default is 5.
    method : str, optional
        Imputation method ('Regularized' or 'EM'). Default is 'Regularized'.
    method_cv : str, optional
        Cross-validation method ('Kfold' or 'loo'). Default is 'Kfold'.
    nbsim : int, optional
        Number of simulations for cross-validation. Default is 100.
    pNA : float, optional
        Proportion of missing values to simulate. Default is 0.05.
    ind_sup : array-like, optional
        Indices of supplementary individuals to exclude from the analysis.
        Indices of supplementary quantitative variables to exclude.
    quali_sup : array-like, optional
        Indices of supplementary qualitative variables to exclude.
    quanti_sup= array-like, optional
        Indices of supplementary quantitative variables to exclude.
    threshold : float, optional
        Convergence threshold. Default is 1e-4.
    verbose : bool, optional
        Whether to print progress. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    dict
        Dictionary containing:
            - 'ncp': Optimal number of dimensions.
            - 'criterion': List of criterion values for each dimension.

    """
    don = don.copy()
    if ind_sup is not None:
        don = don.drop(index=ind_sup)
    if quanti_sup is not None or quali_sup is not None:
        cols_to_drop = []
        if quanti_sup is not None:
            cols_to_drop.extend(don.columns[quanti_sup])
        if quali_sup is not None:
            cols_to_drop.extend(don.columns[quali_sup])
        don = don.drop(columns=cols_to_drop)
    method = method.lower()
    method_cv = method_cv.lower()
    for col in don.columns:
        if not pd.api.types.is_categorical_dtype(don[col]):
            don[col] = don[col].astype("category")
    vrai_tab = tab_disjonctif_NA(don)
    criterion = []
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    if method_cv == "kfold":
        res = np.full((ncp_max - ncp_min + 1, nbsim), np.nan)
        if verbose:
            sim_range = tqdm(range(nbsim), desc="Simulations")
        else:
            sim_range = range(nbsim)
        for sim in sim_range:
            compteur = 0
            max_attempts = 50
            while compteur < max_attempts:
                donNA = prodna(don, pNA, rng)
                categories_complete = all(
                    donNA[col].nunique(dropna=True) == don[col].nunique(dropna=True)  # noqa: E501
                    for col in don.columns
                )
                if categories_complete:
                    break
                compteur += 1
            else:
                raise ValueError(
                    "It is too difficult to suppress some cells.\n"
                    "Maybe several categories are taken by only one individual"
                )
            for nbaxes in range(ncp_min, ncp_max + 1):
                imputed = imputeMCA(
                    donNA,
                    ncp=nbaxes,
                    method=method,
                    threshold=threshold,
                    seed=seed
                )
                tab_disj_comp = imputed["tab_disj"]
                numerator = ((tab_disj_comp - vrai_tab) ** 2).sum().sum()
                denominator = tab_disjonctif_NA(donNA).isna().sum().sum() - vrai_tab.isna().sum().sum()  # noqa: E501
                if denominator == 0:
                    res[nbaxes - ncp_min, sim] = np.nan
                else:
                    res[nbaxes - ncp_min, sim] = numerator / denominator
        crit = np.nanmean(res, axis=1)
        if np.all(np.isnan(crit)):
            raise ValueError("All simulations resulted in NaN error")
        ncp = int(np.nanargmin(crit) + ncp_min)
        criterion = crit.tolist()
        return {"ncp": ncp, "criterion": criterion}


    elif method_cv == "loo":
        criterion = []
        if verbose:
            loop = tqdm(total=(ncp_max - ncp_min + 1) * don.shape[0], desc="LOO CV")  # noqa: E501
        for nbaxes in range(ncp_min, ncp_max + 1):
            errors = []
            for i in range(don.shape[0]):
                donNA = don.copy()
                for col in don.columns:
                    if not pd.isna(donNA.at[donNA.index[i], col]):
                        # Temporarily set the value to NaN
                        donNA.at[donNA.index[i], col] = np.nan
                        # Check if all categories are still represented
                        categories_complete = all(
                            donNA[col].nunique(dropna=True) == don[col].nunique(dropna=True)  # noqa: E501
                            for col in don.columns
                        )
                        if not categories_complete:
                            # Skip this iteration if removing the value causes an issue
                            donNA.at[donNA.index[i], col] = don.at[don.index[i], col]  # noqa: E501
                            continue
                        # Impute missing values using MCA
                        imputed = imputeMCA(
                            donNA,
                            ncp=nbaxes,
                            method=method,
                            threshold=threshold,
                            seed=seed
                        )
                        tab_disj_comp = imputed["tab_disj"]
                        vrai_tab = tab_disjonctif_NA(don)
                        numerator = ((tab_disj_comp - vrai_tab) ** 2).sum().sum()
                        denominator = 1  # Since we imputed one value
                        error = numerator / denominator
                        errors.append(error)
                        # Restore the original value
                        donNA.at[donNA.index[i], col] = don.at[don.index[i], col]
                        if verbose:
                            loop.update(1)
            mean_error = np.mean(errors)
            criterion.append(mean_error)
        if verbose:
            loop.close()
        if np.all(np.isnan(criterion)):
            raise ValueError("All computations resulted in NaN errors")
        ncp = int(np.nanargmin(criterion) + ncp_min)
        return {"ncp": ncp, "criterion": criterion}
    else:
        raise ValueError("method_cv must be 'kfold' or 'loo'")


