import numpy as np  # noqa: D100
import pandas as pd


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
        return 0.0  # or use np.finfo(float).eps for a small positive value
    return np.sum(V[mask] * weights[mask]) / total_weight


def tab_disjonctif_NA(df):
    """Create a disjunctive (one-hot encoded).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with categorical and numeric variables.

    Returns
    -------
    DataFrame
        Disjunctive table with one-hot encoding.

    """  # noqa: E501
    df_encoded_list = []
    for col in df.columns:
        if df[col].dtype.name == "category" or df[col].dtype == object:
            df[col] = df[col].astype("category")
            # Include '__MISSING__' as a category if not already present
            if "__MISSING__" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["__MISSING__"])
            # Fill missing values with '__MISSING__'
            df[col] = df[col].fillna("__MISSING__")
            # One-hot encode the categorical variable
            encoded = pd.get_dummies(
                df[col],
                prefix=col,
                prefix_sep="_",
                dummy_na=False,
                dtype=float,
            )
            df_encoded_list.append(encoded)
        else:
            # Numeric column; keep as is
            df_encoded_list.append(df[[col]])
    # Concatenate all encoded columns
    df_encoded = pd.concat(df_encoded_list, axis=1)
    return df_encoded


def tab_disjonctif_prop(df, seed=None):
    """Perform probabilistic imputation for categorical columns using observed
    value distributions, without creating a separate missing category.

    Parameters
    ----------
    df : DataFrame
        DataFrame with categorical columns to impute.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    DataFrame
        Disjunctive coded DataFrame with missing values probabilistically
        imputed.

    """  # noqa: D205
    if seed is not None:
        np.random.seed(seed)
    df = df.copy()
    df_encoded_list = []
    for col in df.columns:
        if df[col].dtype.name == "category" or df[col].dtype == object:
            # Ensure categories are strings
            df[col] = df[col].cat.rename_categories(
                df[col].cat.categories.astype(str)
            )
            observed = df[col][df[col].notna()]
            categories = df[col].cat.categories.tolist()
            # Get observed frequencies
            freqs = observed.value_counts(normalize=True)
            # Impute missing values based on observed frequencies
            missing_indices = df[col][df[col].isna()].index
            if len(missing_indices) > 0:
                imputed_values = np.random.choice(
                    freqs.index, size=len(missing_indices), p=freqs.values
                )
                df.loc[missing_indices, col] = imputed_values
            # One-hot encode without creating missing category
            encoded = pd.get_dummies(
                df[col],
                prefix=col,
                prefix_sep="_",
                dummy_na=False,
                dtype=float,
            )
            col_names = [f"{col}_{cat}" for cat in categories]
            encoded = encoded.reindex(columns=col_names, fill_value=0.0)
            df_encoded_list.append(encoded)
        else:
            df_encoded_list.append(df[[col]])
    df_encoded = pd.concat(df_encoded_list, axis=1)
    return df_encoded


def find_category(df_original, tab_disj):
    """Reconstruct the original categorical variables from the disjunctive.

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
            if "__MISSING__" in categories:
                missing_cat_index = categories.index("__MISSING__")
            else:
                missing_cat_index = None
            num_categories = len(categories)
            sub_tab = tab_disj.iloc[:, start_idx : start_idx + num_categories]
            if missing_cat_index is not None:
                sub_tab.iloc[:, missing_cat_index] = -np.inf
            # Find the category with the maximum value for each row
            max_indices = sub_tab.values.argmax(axis=1)
            df_reconstructed[col] = [categories[idx] for idx in max_indices]
            # Replace '__MISSING__' back to NaN
            df_reconstructed[col].replace("__MISSING__", np.nan, inplace=True)
            start_idx += num_categories
        else:
            # For numeric variables, keep as is
            start_idx += 1  # Increment start_idx by 1 for numeric columns
    return df_reconstructed


def svdtriplet(X, row_w=None, ncp=np.inf):
    """Perform weighted SVD on matrix X with row weights.

    Parameters
    ----------
    X : ndarray
        Data matrix of shape (n_samples, n_features).
    row_w : array-like, optional
        Row weights. If None, uniform weights are assumed. Default is None.
    ncp : int
        Number of principal components to retain. Default is infinity.

    Returns
    -------
    s : ndarray
        Singular values.
    U : ndarray
        Left singular vectors.
    V : ndarray
        Right singular vectors.

    """
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=float)
    else:
        X = X.astype(float)
    if row_w is None:
        row_w = np.ones(X.shape[0]) / X.shape[0]
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()
    ncp = int(min(ncp, X.shape[0] - 1, X.shape[1]))
    # Apply weights to rows
    X_weighted = X * np.sqrt(row_w[:, None])
    # Perform SVD
    U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)
    V = Vt.T
    U = U[:, :ncp]
    V = V[:, :ncp]
    s = s[:ncp]
    # Adjust signs to ensure consistency
    mult = np.sign(np.sum(V, axis=0))
    mult[mult == 0] = 1
    U *= mult
    V *= mult
    # Rescale U by the square root of row weights
    U /= np.sqrt(row_w[:, None])
    return s, U, V


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
    # Ensure the data is a DataFrame
    don = pd.DataFrame(don)
    don = don.copy()

    for col in don.columns:
        if (
            not pd.api.types.is_numeric_dtype(don[col])
            or don[col].dtype == "bool"
        ):  # noqa: E501
            don[col] = don[col].astype("category")
            # Convert categories to strings and rename them
            new_categories = don[col].cat.categories.astype(str)
            don[col] = don[col].cat.rename_categories(new_categories)
        else:
            unique_values = don[col].dropna().unique()
            if set(unique_values).issubset({0, 1}):
                don[col] = don[col].astype("category")
                new_categories = don[col].cat.categories.astype(str)
                don[col] = don[col].cat.rename_categories(new_categories)  # noqa: E501

    print("Data types after conversion:")
    print(don.dtypes)

    # Handle row weights
    if row_w is None:
        row_w = np.ones(len(don)) / len(don)
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()

    # Initial imputation and creation of disjunctive tables
    tab_disj_NA = tab_disjonctif_NA(don)
    tab_disj_comp = tab_disjonctif_prop(don, seed=seed)
    hidden = tab_disj_NA.isna()
    tab_disj_rec_old = tab_disj_comp.copy()

    # Initialize iteration parameters
    nbiter = 0
    continue_flag = True

    while continue_flag:
        nbiter += 1

        # Step 1: Compute weighted means M
        M = (
            tab_disj_comp.apply(lambda col: moy_p(col.values, row_w))
            / don.shape[1]
        )  # noqa: E501
        M = M.replace({0: np.finfo(float).eps})
        M = M.fillna(np.finfo(float).eps)

        if (M < 0).any():
            raise ValueError(
                "Negative values encountered in M. Check data preprocessing."
            )  # noqa: E501

        print(f"Iteration {nbiter}:")
        print("Weighted means (M):")
        print(M.head())

        # Step 2: Center and scale the data
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

        print("Centered and scaled data (Zscale):")
        print(Zscale.head())

        # Step 3: Perform weighted SVD
        s, U, V = svdtriplet(Zscale.values, row_w=row_w, ncp=ncp)
        print("Singular values (s):")
        print(s)
        print("Left singular vectors (U):")
        print(U)
        print("Right singular vectors (V):")
        print(V)

        # Step 4: Regularization (Shrinking Eigenvalues)
        if method.lower() == "em":
            moyeig = 0
        else:
            # Calculate moyeig based on R's imputeMCA logic
            if len(s) > ncp:
                moyeig = np.mean(s[ncp:] ** 2)
                moyeig = min(moyeig * coeff_ridge, s[ncp] ** 2)
            else:
                moyeig = 0
                # Set to 0 when there are no additional singular values
        eig_shrunk = (s[:ncp] ** 2 - moyeig) / s[:ncp]
        eig_shrunk = np.maximum(eig_shrunk, 0)  # Ensure non-negative
        print("Shrunk eigenvalues (eig_shrunk):")
        print(eig_shrunk)

        # Step 5: Reconstruct the data
        rec = U @ np.diag(eig_shrunk) @ V.T
        tab_disj_rec = pd.DataFrame(
            rec, columns=tab_disj_comp.columns, index=tab_disj_comp.index
        )  # noqa: E501
        tab_disj_rec = tab_disj_rec.div(np.sqrt(M), axis=1) + 1
        tab_disj_rec = tab_disj_rec.multiply(tab_disj_comp_mean, axis=1)
        print("Reconstructed disjunctive table (tab_disj_rec):")
        print(tab_disj_rec.head())

        # Step 6: Compute difference and relative change
        diff = tab_disj_rec - tab_disj_rec_old
        diff_values = diff.values
        hidden_values = hidden.values
        # Zero out observed positions
        diff_values[~hidden_values] = 0
        relch = np.sum((diff_values**2) * row_w[:, None])
        print(f"Relative Change: {relch}\n")

        # Step 7: Update for next iteration
        tab_disj_rec_old = tab_disj_rec.copy()
        tab_disj_comp.values[hidden_values] = tab_disj_rec.values[
            hidden_values
        ]  # noqa: E501

        # Step 8: Check convergence
        continue_flag = (relch > threshold) and (nbiter < maxiter)

    # Step 9: Reconstruct categorical data
    completeObs = find_category(don, tab_disj_comp)

    return {"tab_disj": tab_disj_comp, "completeObs": completeObs}
