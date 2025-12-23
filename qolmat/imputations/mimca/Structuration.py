from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Optional, Dict, Any

# --- Helper functions ---

def moy_p(V: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean of non-NaN elements in V."""
    mask = ~np.isnan(V)
    total_weight = np.sum(weights[mask])
    if total_weight == 0:
        return 0.0
    return np.sum(V[mask] * weights[mask]) / total_weight

def tab_disjonctif_NA(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a disjunctive (one-hot encoded) table from a DataFrame,
    preserving NaN values.
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
    return pd.concat(df_encoded_list, axis=1)

def tab_disjonctif_prop(df: pd.DataFrame, seed: Optional[int] = None, row_w: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Initialize missing values in the disjunctive table by replacing NaNs
    with the weighted column means.
    """
    tab = tab_disjonctif_NA(df)
    if row_w is None:
        row_w = np.ones(len(df)) / len(df)
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()
    init_vals = tab.apply(lambda col: moy_p(col.values, row_w))
    return tab.fillna(init_vals)

def find_category(df_original: pd.DataFrame, tab_disj: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the original categorical variables from the imputed disjunctive table.
    """
    df_reconstructed = df_original.copy()
    start_idx = 0
    for col in df_original.columns:
        if df_original[col].dtype.name == "category" or df_original[col].dtype == object:
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

def svdtriplet(X: np.ndarray, row_w: Optional[np.ndarray] = None, ncp: Union[int, float] = np.inf) -> tuple:
    """
    Perform weighted SVD on matrix X using row weights.
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
    X_weighted = X * np.sqrt(row_w[:, None])
    U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)
    V = Vt.T
    U = U[:, :ncp]
    V = V[:, :ncp]
    s = s[:ncp]
    mult = np.sign(np.sum(V, axis=0))
    mult[mult == 0] = 1
    U *= mult
    V *= mult
    U /= np.sqrt(row_w[:, None])
    return s, U, V

# --- MIMCA Class ---

class MIMCA(BaseEstimator, TransformerMixin):
    """
    Multiple Imputation using Multiple Correspondence Analysis (MCA)
    in a scikit-learn compatible estimator.
    
    Parameters
    ----------
    ncp : int, default=2
        Number of principal components to retain.
    method : str, default="Regularized"
        Imputation method. Either "Regularized" or "EM".
    row_w : Optional[np.ndarray], default=None
        Row weights. If None, uniform weights are used.
    coeff_ridge : float, default=1
        Regularization coefficient for Regularized MCA.
    threshold : float, default=1e-6
        Convergence threshold.
    maxiter : int, default=1000
        Maximum number of iterations.
    debug : bool, default=False
        If True, print internal debug information.
    """
    
    def __init__(
        self,
        ncp: int = 2,
        method: str = "Regularized",
        row_w: Optional[np.ndarray] = None,
        coeff_ridge: float = 1,
        threshold: float = 1e-6,
        maxiter: int = 1000,
        debug: bool = False,
    ) -> None:
        self.ncp = ncp
        self.method = method
        self.row_w = row_w
        self.coeff_ridge = coeff_ridge
        self.threshold = threshold
        self.maxiter = maxiter
        self.debug = debug

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> MIMCA:
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by imputing missing values using MCA.
        
        Returns
        -------
        pd.DataFrame
            The complete dataset with missing values imputed.
        """
        res = imputeMCA(
            X,
            ncp=self.ncp,
            method=self.method,
            row_w=self.row_w,
            coeff_ridge=self.coeff_ridge,
            threshold=self.threshold,
            maxiter=self.maxiter,
            debug=self.debug,
        )
        return res["completeObs"]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Fit and transform the data.
        """
        return self.fit(X, y).transform(X)

    def impute_indicator(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return the imputed disjunctive (indicator) matrix.
        """
        res = imputeMCA(
            X,
            ncp=self.ncp,
            method=self.method,
            row_w=self.row_w,
            coeff_ridge=self.coeff_ridge,
            threshold=self.threshold,
            maxiter=self.maxiter,
            debug=self.debug,
        )
        return res["tab_disj"]

    def estimate_ncp(
        self,
        X: pd.DataFrame,
        ncp_min: int = 0,
        ncp_max: int = 5,
        method_cv: str = "Kfold",
        nbsim: int = 100,
        pNA: float = 0.05,
        ind_sup: Optional[np.ndarray] = None,
        quanti_sup: Optional[np.ndarray] = None,
        quali_sup: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate the optimal number of components (ncp) using cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        ncp_min : int, default=0
            Minimum number of components to test.
        ncp_max : int, default=5
            Maximum number of components to test.
        method_cv : str, default="Kfold"
            Cross-validation method: "Kfold" or "loo".
        nbsim : int, default=100
            Number of simulations (for Kfold).
        pNA : float, default=0.05
            Proportion of missing values to simulate.
        ind_sup, quanti_sup, quali_sup : Optional[np.ndarray]
            Indices of supplementary individuals or variables (if any).
        seed : Optional[int]
            Random seed.
        verbose : bool, default=True
            If True, show progress.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
              - "ncp": optimal number of components.
              - "criterion": list of CV error values for each tested ncp.
        """
        return estim_ncpMCA(
            X,
            ncp_min=ncp_min,
            ncp_max=ncp_max,
            method=self.method,
            method_cv=method_cv,
            nbsim=nbsim,
            pNA=pNA,
            ind_sup=ind_sup,
            quanti_sup=quanti_sup,
            quali_sup=quali_sup,
            threshold=self.threshold,
            verbose=verbose,
            seed=seed
        )

# --- Core imputation function (with debug prints) ---

def imputeMCA(
    don: pd.DataFrame,
    ncp: int = 2,
    method: str = "Regularized",
    row_w: Optional[np.ndarray] = None,
    coeff_ridge: float = 1,
    threshold: float = 1e-6,
    seed: Optional[int] = None,
    maxiter: int = 1000,
    debug: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Impute missing values in a dataset using MCA.
    
    Parameters
    ----------
    don : pd.DataFrame
        Input dataset with missing values.
    ncp : int, default=2
        Number of principal components.
    method : str, default="Regularized"
        Either "Regularized" or "EM".
    row_w : Optional[np.ndarray], default=None
        Row weights; if None, uniform weights are used.
    coeff_ridge : float, default=1
        Regularization coefficient.
    threshold : float, default=1e-6
        Convergence threshold.
    seed : Optional[int]
        Random seed.
    maxiter : int, default=1000
        Maximum iterations.
    debug : bool, default=False
        If True, print debug information.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with:
          - "tab_disj": the imputed disjunctive table.
          - "completeObs": the reconstructed categorical DataFrame.
    """
    don = pd.DataFrame(don).copy()
    # Convert non-numeric columns to categorical
    for col in don.columns:
        if not pd.api.types.is_numeric_dtype(don[col]) or don[col].dtype == "bool":
            don[col] = don[col].astype("category")
            new_categories = don[col].cat.categories.astype(str)
            don[col] = don[col].cat.rename_categories(new_categories)
        else:
            unique_vals = don[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                don[col] = don[col].astype("category")
                new_categories = don[col].cat.categories.astype(str)
                don[col] = don[col].cat.rename_categories(new_categories)
    if row_w is None:
        row_w = np.ones(len(don)) / len(don)
    else:
        row_w = np.array(row_w, dtype=float)
        row_w /= row_w.sum()
    
    tab_disj_NA_df = tab_disjonctif_NA(don)
    tab_disj_comp = tab_disjonctif_prop(don, seed=seed, row_w=row_w)
    hidden = tab_disj_NA_df.isna()  # mask for missing cells
    tab_disj_rec_old = tab_disj_comp.copy()
    method = method.lower()
    nbiter = 0
    continue_flag = True
    
    while continue_flag:
        nbiter += 1
        # Compute weighted column means normalized by number of original variables.
        M = tab_disj_comp.apply(lambda col: moy_p(col.values, row_w)) / don.shape[1]
        M = M.replace(0, np.finfo(float).eps).fillna(np.finfo(float).eps)
        col_means = tab_disj_comp.apply(lambda col: moy_p(col.values, row_w))
        col_means = col_means.replace(0, np.finfo(float).eps)
        # Center and scale.
        Z = tab_disj_comp.div(col_means, axis=1)
        Z = Z.subtract(Z.apply(lambda col: moy_p(col.values, row_w), axis=0), axis=1)
        Zscale = Z.multiply(np.sqrt(M), axis=1)
        
        # Full SVD on Zscale.
        s_full, U_full, V_full = np.linalg.svd(Zscale.values, full_matrices=False)
        s_full = np.array(s_full)
        
        if method == "em":
            moyeig = 0
        else:
            if len(s_full) > ncp:
                tail_vals = s_full[ncp:]
                moyeig = np.mean(tail_vals**2) if tail_vals.size > 0 else 0
                moyeig = min(float(moyeig * coeff_ridge), float((s_full[ncp]**2).item()))
            else:
                moyeig = 0
        
        if debug:
            print(f"Iteration {nbiter}")
            print("s_full:", s_full)
            print("Selected singular values (first ncp):", s_full[:ncp])
            print("Computed moyeig:", moyeig)
        
        U = U_full[:, :ncp]
        V = V_full[:ncp, :].T
        lambda_vals = s_full[:ncp]
        eig_shrunk = (lambda_vals**2 - moyeig) / (lambda_vals + 1e-15)
        eig_shrunk = np.maximum(eig_shrunk, 0)
        
        if ncp > 0:
            rec = U @ np.diag(eig_shrunk) @ V.T
        else:
            rec = np.zeros_like(Zscale.values)
        
        tab_disj_rec = pd.DataFrame(rec, index=tab_disj_comp.index, columns=tab_disj_comp.columns)
        # Reverse scaling and centering.
        tab_disj_rec = tab_disj_rec.div(np.sqrt(M), axis=1).add(1.0)
        tab_disj_rec = tab_disj_rec.multiply(col_means, axis=1)
        
        diff = tab_disj_rec - tab_disj_rec_old
        diff.values[~hidden.values] = 0
        rel_change = np.sum((diff.values**2) * row_w[:, None])
        
        if debug:
            print("Relative change:", rel_change)
        
        tab_disj_comp.values[hidden.values] = tab_disj_rec.values[hidden.values]
        tab_disj_rec_old = tab_disj_rec.copy()
        continue_flag = (rel_change > threshold) and (nbiter < maxiter)
    
    completeObs = find_category(don, tab_disj_comp)
    if debug:
        print("Converged after", nbiter, "iterations")
    return {"tab_disj": tab_disj_comp, "completeObs": completeObs}

# --- CV Estimation Function ---

def estim_ncpMCA(
    don: pd.DataFrame,
    ncp_min: int = 0,
    ncp_max: int = 5,
    method: str = "Regularized",
    method_cv: str = "Kfold",
    nbsim: int = 100,
    pNA: float = 0.05,
    ind_sup: Optional[np.ndarray] = None,
    quanti_sup: Optional[np.ndarray] = None,
    quali_sup: Optional[np.ndarray] = None,
    threshold: float = 1e-4,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate the optimal number of components (ncp) for MCA using cross-validation.
    
    Returns a dictionary with:
      - "ncp": optimal number of components.
      - "criterion": list of CV error values for each tested ncp.
    """
    don = don.copy()
    if ind_sup is not None:
        don = don.drop(index=ind_sup)
    if quanti_sup is not None or quali_sup is not None:
        cols_to_drop = []
        if quanti_sup is not None:
            cols_to_drop.extend(don.columns[list(quanti_sup)])
        if quali_sup is not None:
            cols_to_drop.extend(don.columns[list(quali_sup)])
        don = don.drop(columns=cols_to_drop)
    for col in don.columns:
        if not pd.api.types.is_categorical_dtype(don[col]):
            don[col] = don[col].astype("category")
    vrai_tab = tab_disjonctif_NA(don)
    criterion = []
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    if method_cv.lower() == "kfold":
        res = np.full((ncp_max - ncp_min + 1, nbsim), np.nan)
        for sim in range(nbsim):
            max_attempts = 50
            for attempt in range(max_attempts):
                donNA = don.copy()
                total_cells = donNA.shape[0] * donNA.shape[1]
                n_missing = int(np.floor(total_cells * pNA))
                idx = rng.choice(total_cells, n_missing, replace=False)
                row_idx = idx // donNA.shape[1]
                col_idx = idx % donNA.shape[1]
                for r, c in zip(row_idx, col_idx):
                    donNA.iat[r, c] = np.nan
                if all(donNA[col].nunique(dropna=True) == don[col].nunique(dropna=True) for col in don.columns):
                    break
            else:
                raise ValueError("Too many attempts to inject missing values without dropping categories.")
            for nb in range(ncp_min, ncp_max + 1):
                imputed = imputeMCA(donNA, ncp=nb, method=method, threshold=threshold, seed=seed)
                tab_comp = imputed["tab_disj"]
                numerator = ((tab_comp - vrai_tab)**2).sum().sum()
                denom = tab_disjonctif_NA(donNA).isna().sum().sum() - vrai_tab.isna().sum().sum()
                res[nb - ncp_min, sim] = numerator / denom if denom != 0 else np.nan
        crit = np.nanmean(res, axis=1)
        if np.all(np.isnan(crit)):
            raise ValueError("All simulations resulted in NaN error")
        opt_ncp = int(np.nanargmin(crit) + ncp_min)
        criterion = crit.tolist()
        return {"ncp": opt_ncp, "criterion": criterion}
    elif method_cv.lower() == "loo":
        criterion = []
        for nb in range(ncp_min, ncp_max + 1):
            errors = []
            for i in range(don.shape[0]):
                for col in don.columns:
                    if not pd.isna(don.at[don.index[i], col]):
                        donNA = don.copy()
                        donNA.at[don.index[i], col] = np.nan
                        if donNA[col].nunique(dropna=True) < don[col].nunique(dropna=True):
                            continue
                        imputed = imputeMCA(donNA, ncp=nb, method=method, threshold=threshold, seed=seed)
                        tab_comp = imputed["tab_disj"]
                        diff = (tab_comp - vrai_tab)**2
                        errors.append(diff.to_numpy().sum())
            mean_err = np.nan if len(errors) == 0 else np.mean(errors)
            criterion.append(mean_err)
        if np.all(np.isnan(criterion)):
            raise ValueError("All computations resulted in NaN errors")
        opt_ncp = int(np.nanargmin(criterion) + ncp_min)
        return {"ncp": opt_ncp, "criterion": criterion}
    else:
        raise ValueError("method_cv must be 'Kfold' or 'loo'")
