import numpy as np  # noqa: D100
import pandas as pd

from qolmat.utils.algebra import svdtriplet
from qolmat.utils.utils import (
    find_category,
    moy_p,
    tab_disjonctif_NA,
    tab_disjonctif_prop,
)


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
