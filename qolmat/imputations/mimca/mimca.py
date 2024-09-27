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


def tab_disjonctif_NA(df) -> pd.DataFrame:
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
        if (
            df_original[col].dtype.name == "category"
            or df_original[col].dtype == object
        ):  # noqa: E501
            categories = df_original[col].cat.categories.tolist()
            num_categories = len(categories)
            sub_tab = tab_disj.iloc[:, start_idx : start_idx + num_categories]
            max_indices = sub_tab.values.argmax(axis=1)
            df_reconstructed[col] = [categories[idx] for idx in max_indices]
            df_reconstructed[col] = df_reconstructed[col].astype("category")
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
    """Impute missing values in a dataset using (MCA).

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
        if (
            not pd.api.types.is_numeric_dtype(don[col])
            or don[col].dtype == "bool"
        ):  # noqa: E501
            don[col] = don[col].astype("category")
            new_categories = don[col].cat.categories.astype(str)
            don[col] = don[col].cat.rename_categories(new_categories)  # noqa: E501
        else:
            unique_values = don[col].dropna().unique()
            if set(unique_values).issubset({0, 1}):
                don[col] = don[col].astype("category")
                new_categories = don[col].cat.categories.astype(str)
                don[col] = don[col].cat.rename_categories(new_categories)  # noqa: E501
    if row_w is None:
        row_w = np.ones(len(don)) / len(don)
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()
    tab_disj_NA = tab_disjonctif_NA(don)
    if ncp == 0:
        tab_disj_comp_mean = tab_disj_NA.apply(
            lambda col: moy_p(col.values, row_w)
        )  # noqa: E501
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
        M = (
            tab_disj_comp.apply(lambda col: moy_p(col.values, row_w))
            / don.shape[1]
        )  # noqa: E501
        M = M.replace({0: np.finfo(float).eps})
        M = M.fillna(np.finfo(float).eps)
        tab_disj_comp_mean = tab_disj_comp.apply(
            lambda col: moy_p(col.values, row_w)
        )  # noqa: E501
        tab_disj_comp_mean = tab_disj_comp_mean.replace(
            {0: np.finfo(float).eps}
        )  # noqa: E501
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
        eig_shrunk = (s**2 - moyeig) / s
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
        tab_disj_comp.values[hidden_values] = tab_disj_rec.values[
            hidden_values
        ]  # noqa: E501
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
    seed=None,
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
    quanti_sup : array-like, optional
        Indices of supplementary quantitative variables to exclude.
    quali_sup : array-like, optional
        Indices of supplementary qualitative variables to exclude.
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
            - 'criterion': List of criterion values dimensions.

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
                    donNA[col].nunique(dropna=True)
                    == don[col].nunique(dropna=True)  # noqa: E501
                    for col in don.columns
                )
                if categories_complete:
                    break
                compteur += 1
            else:
                raise ValueError(
                    "It is too difficult to suppress some cells.\n"
                    "Maybe several categories by only one individual. "
                    'You should remove these variables or try with"loo".'
                )
            for nbaxes in range(ncp_min, ncp_max + 1):
                imputed = imputeMCA(
                    donNA,
                    ncp=nbaxes,
                    method=method,
                    threshold=threshold,
                    seed=seed,
                )
                tab_disj_comp = imputed["tab_disj"]
                numerator = ((tab_disj_comp - vrai_tab) ** 2).sum().sum()
                denominator = (
                    tab_disjonctif_NA(donNA).isna().sum().sum()
                    - vrai_tab.isna().sum().sum()
                )  # noqa: E501
                if denominator == 0:
                    res[nbaxes - ncp_min, sim] = np.nan
                else:
                    res[nbaxes - ncp_min, sim] = numerator / denominator
        crit = np.nanmean(res, axis=1)
        if np.all(np.isnan(crit)):
            raise ValueError(
                "All simulations resulted in NaN errors. Please check your data and parameters."
            )  # noqa: E501
        ncp = int(np.nanargmin(crit) + ncp_min)
        criterion = crit.tolist()
        return {"ncp": ncp, "criterion": criterion}
    elif method_cv == "loo":
        # LOO cross-validation code (if needed)
        pass
    else:
        raise ValueError("method_cv must be 'kfold' or 'loo'")


def imputeMCA_print(
    don,
    ncp,
    method="Regularized",
    row_w=None,
    coeff_ridge=1,
    threshold=1e-6,
    seed=None,
    maxiter=1000,
    verbose=False,
    print_msg="",
):
    """Print progress during MCA imputation.

    Parameters
    ----------
    don : DataFrame
        Input dataset with missing values.
    ncp : int
        Number of principal components for MCA.
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
    verbose : bool, optional
        Whether to print progress. Default is False.
    print_msg : str, optional
        Message to print during imputation. Default is ''.

    Returns
    -------
    dict
        Result of the MCA imputation.

    """
    if verbose:
        print(f"{print_msg}...", end="", flush=True)
    res = imputeMCA(
        don=don,
        ncp=ncp,
        method=method,
        row_w=row_w,
        coeff_ridge=coeff_ridge,
        threshold=threshold,
        seed=seed,
        maxiter=maxiter,
    )  # noqa: E501
    if verbose:
        print("done")
    return res


def normtdc(tab_disj, data_na):
    """Normalize the disjunctive table to ensure values are between 0 and 1.

    Parameters
    ----------
    tab_disj : DataFrame
        Disjunctive table to normalize.
    data_na : DataFrame
        DataFrame with original categorical data.

    Returns
    -------
    DataFrame
        Normalized disjunctive table.

    """
    tdc = tab_disj.copy()
    tdc[tdc < 0] = 0
    tdc[tdc > 1] = 1
    col_suppr = np.cumsum(
        [len(col.cat.categories) for _, col in data_na.items()]
    )  # noqa: E501

    def normalize_row(row, col_suppr):
        start = 0
        for end in col_suppr:
            segment = row[start:end]
            total = np.sum(segment)
            if total != 0:
                row[start:end] = segment / total
            start = end
        return row

    tdc = tdc.apply(
        lambda row: normalize_row(row.values, col_suppr),
        axis=1,
        result_type="expand",
    )  # noqa: E501
    tdc.columns = tab_disj.columns
    return tdc


def draw(tab_disj, Don, Don_na):
    """Draw random samples from the normalized disjtable to reconstruct data.

    Parameters
    ----------
    tab_disj : DataFrame
        Normalized disjunctive table.
    Don : DataFrame
        Original complete dataset.
    Don_na : DataFrame
        Dataset with missing values.

    Returns
    -------
    DataFrame
        Reconstructed dataset with imputed categorical values.

    """
    Don_res = Don.copy()
    nbdummy = np.ones(Don.shape[1], dtype=int)
    is_quali = [
        i
        for i, col in enumerate(Don.columns)
        if not pd.api.types.is_numeric_dtype(Don[col])
    ]  # noqa: E501
    nbdummy[is_quali] = [Don.iloc[:, i].nunique() for i in is_quali]
    vec = np.concatenate(([0], np.cumsum(nbdummy)))
    for idx, i in enumerate(is_quali):
        start = vec[idx]
        end = vec[idx + 1]
        cols = tab_disj.columns[start:end]
        probs = tab_disj[cols].values
        categories = Don.iloc[:, i].cat.categories
        sampled_indices = []
        for p in probs:
            if np.sum(p) > 0:
                p_normalized = p / np.sum(p)
                sampled_idx = np.random.choice(len(categories), p=p_normalized)  # noqa: E501
            else:
                sampled_idx = np.nan
            sampled_indices.append(sampled_idx)
        Don_res.iloc[:, i] = pd.Categorical.from_codes(
            sampled_indices, categories=categories
        )  # noqa: E501
    return Don_res


def MIMCA(
    X,
    nboot=100,
    ncp=2,
    coeff_ridge=1,
    threshold=1e-6,
    maxiter=1000,
    verbose=False,
):  # noqa: E501
    """Perform Multiple Imputation with (MIMCA).

    Parameters
    ----------
    X : DataFrame
        Input data with missing values.
    nboot : int, optional
        Number of bootstrap samples. Default is 100.
    ncp : int, optional
        Number of principal components for MCA. Default is 2.
    coeff_ridge : float, optional
        Regularization coefficient for 'Regularized' MCA. Default is 1.
    threshold : float, optional
        Convergence threshold. Default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations for the imputation process.
    verbose : bool, optional
        Whether to print progress. Default is False.

    Returns
    -------
    dict
        Dictionary containing the results of the multiple imputations.

    """
    import warnings

    X = X.copy()
    # Convert non-numeric columns to categorical
    is_quali = [
        col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
    ]  # noqa: E501
    X[is_quali] = X[is_quali].apply(lambda col: col.astype("category"))
    X = X.apply(
        lambda col: col.cat.remove_unused_categories()
        if col.dtype.name == "category"
        else col
    )  # noqa: E501
    # Remove variables with only one category
    OneCat = (
        X.apply(
            lambda col: len(col.cat.categories)
            if col.dtype.name == "category"
            else np.nan
        )
        == 1
    )  # noqa: E501
    if OneCat.any():
        warning_vars = X.columns[OneCat].tolist()
        warnings.warn(
            f"The following variables are constant and have been suppressed from the analysis: {', '.join(warning_vars)}"
        )  # noqa: E501
        X = X.drop(columns=warning_vars)
        if X.shape[1] <= 1:
            raise ValueError(
                "No sufficient variables have 2 categories or more"
            )  # noqa: E501
    n = X.shape[0]
    # Generate bootstrap weights
    rng = np.random.default_rng()
    Boot = rng.integers(low=0, high=n, size=(n, nboot))
    Weight = np.zeros((n, nboot))
    for i in range(nboot):
        counts = np.bincount(Boot[:, i], minlength=n)
        Weight[:, i] = counts
    Weight = Weight / Weight.sum(axis=0)
    # Perform multiple imputations
    res_imp = []
    for i in range(nboot):
        if verbose:
            print(f"Imputation {i + 1}/{nboot}")
        weight_i = Weight[:, i]
        res = imputeMCA_print(
            don=X,
            ncp=ncp,
            coeff_ridge=coeff_ridge,
            threshold=threshold,  # noqa: E501
            maxiter=maxiter,
            row_w=weight_i,
            verbose=verbose,
            print_msg=f"Imputation {i + 1}",
        )  # noqa: E501
        res_imp.append(res)
    # Normalize the imputed disjunctive tables
    tdc_imp = [res["tab_disj"] for res in res_imp]
    res_comp = [res["completeObs"] for res in res_imp]
    tdc_norm = [
        normtdc(tab_disj=tdc, data_na=comp)
        for tdc, comp in zip(tdc_imp, res_comp)
    ]  # noqa: E501
    # Draw the final imputed datasets
    X_imp = [
        draw(tab_disj=tdc, Don=comp, Don_na=X)
        for tdc, comp in zip(tdc_norm, res_comp)
    ]  # noqa: E501
    # Compute the final imputed disjunctive table using all data
    res_imputeMCA = imputeMCA(
        X,
        ncp=ncp,
        coeff_ridge=coeff_ridge,
        threshold=threshold,
        maxiter=maxiter,
    )["tab_disj"]
    res = {
        "res_MIs": X_imp,
        "res_imputeMCA": res_imputeMCA,
        "call": {
            "X": X,
            "nboot": nboot,
            "ncp": ncp,
            "coeff_ridge": coeff_ridge,
            "threshold": threshold,
            "maxiter": maxiter,
            "tab_disj_array": np.array([tdc.values for tdc in tdc_imp]),
        },
    }
    return res
