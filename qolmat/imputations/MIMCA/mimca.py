import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _total_variance_distance_1D(df1: pd.Series, df2: pd.Series) -> float:
    """Compute Total Variance Distance for a categorical feature."""
    try:
        list_categories = list(set(df1.unique()).union(set(df2.unique())))
        freqs1 = df1.value_counts() / len(df1)
        freqs1 = freqs1.reindex(list_categories, fill_value=0.0)
        freqs2 = df2.value_counts() / len(df2)
        freqs2 = freqs2.reindex(list_categories, fill_value=0.0)
        return (freqs1 - freqs2).abs().sum()
    except Exception as e:
        logging.error(f"Error computing TVD: {e}")
        return np.nan

def create_disjunctive_table(X):
    """Create a disjunctive table from categorical columns."""
    try:
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        dummies = pd.get_dummies(X[categorical_columns], dummy_na=False, dtype=float)
        for col in categorical_columns:
            na_rows = X[col].isna()
            new_cols = [new_col for new_col in dummies.columns if new_col.startswith(col + '_')]
            dummies.loc[na_rows, new_cols] = np.nan
        return dummies
    except Exception as e:
        logging.error(f"Error creating disjunctive table: {e}")
        raise

def initialize_missing_values(Z):
    """Initialize missing values in the disjunctive table with column proportions."""
    try:
        proportions = Z.mean(axis=0)
        return Z.fillna(proportions)
    except Exception as e:
        logging.error(f"Error initializing missing values: {e}")
        raise

def combine_estimates(estimates, variances):
    """Combine multiple imputation estimates using Rubin's rules."""
    try:
        M = len(estimates)
        psi_hat = np.mean(estimates, axis=0)
        within_variance = np.mean(variances, axis=0)
        between_variance = np.var(estimates, axis=0, ddof=1)
        total_variance = within_variance + (1 + 1/M) * between_variance
        return psi_hat, total_variance
    except Exception as e:
        logging.error(f"Error combining estimates: {e}")
        raise

def compute_confidence_interval(psi_hat, total_variance, M, alpha=0.05):
    """Compute confidence interval for combined estimate."""
    try:
        t_value = 1.96  # for 95% confidence interval
        margin_error = t_value * np.sqrt(total_variance)
        lower_bound = psi_hat - margin_error
        upper_bound = psi_hat + margin_error
        return lower_bound, upper_bound
    except Exception as e:
        logging.error(f"Error computing confidence interval: {e}")
        raise

class MIMCA:
    """Multiple Imputation using Multiple Correspondence Analysis (MIMCA)."""

    def __init__(self, n_components=2, max_iter=10, tolerance=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

    def perform_mca(self, Z, R):
        """Perform Multiple Correspondence Analysis (MCA) on the disjunctive table."""
        try:
            N_all = np.sum(Z.values)
            Z_normalized = Z / N_all
            Sum_r = np.sum(Z_normalized, axis=1)
            Sum_c = np.sum(Z_normalized, axis=0)
            Z_expected = np.outer(Sum_r, Sum_c)
            Z_residual = Z_normalized - Z_expected
            D_r_sqrt_mi = np.diag(1 / np.sqrt(Sum_r + 1e-10))
            D_c_sqrt_mi = np.diag(1 / np.sqrt(Sum_c + 1e-10))
            MCA_mat = D_r_sqrt_mi @ Z_residual.values @ D_c_sqrt_mi
            U, s, Vt = np.linalg.svd(MCA_mat)
            return U, s, Vt
        except Exception as e:
            logging.error(f"Error performing MCA: {e}")
            raise

    def fit(self, X, M=5):
        """Fit the MIMCA model to the data."""
        try:
            Z = create_disjunctive_table(X)
            W = (~np.isnan(Z)).astype(int)
            Z = initialize_missing_values(Z)

            imputed_datasets = []
            estimates = []
            variances = []

            for m in range(M):
                # Step 1: Reflect the variability on the set of parameters of the imputation model
                indices = np.random.choice(Z.shape[0], size=Z.shape[0], replace=True)
                Rboot = np.bincount(indices, minlength=Z.shape[0])

                for _ in range(self.max_iter):
                    # Step 2a: Perform MCA
                    Z_old = Z.copy()
                    U, s, Vt = self.perform_mca(Z, Rboot)

                    Lambda_half = np.diag(np.sqrt(s[:self.n_components]))
                    U_s = U[:, :self.n_components]
                    Vt_s = Vt[:self.n_components, :]

                    Z_fitted_centered = U_s @ Lambda_half @ Vt_s
                    Z_fitted = pd.DataFrame(Z_fitted_centered + Z.mean(axis=0).values, columns=Z.columns)

                    # Ensure imputed values are normalized by category
                    for col in X.select_dtypes(include=['object', 'category']).columns:
                        col_pattern = f"{col}_"
                        col_mask = Z.columns.str.startswith(col_pattern)
                        Z_fitted.loc[:, col_mask] = Z_fitted.loc[:, col_mask].div(Z_fitted.loc[:, col_mask].sum(axis=1), axis=0)

                    # Step 2b: Impute the missing values
                    Z_new = W * Z_old + (1 - W) * Z_fitted

                    # Step 2c: Check for convergence
                    if np.linalg.norm(Z_new - Z_old, 'fro') < self.tolerance:
                        break

                    Z = Z_new

                estimates.append(Z.mean().values)
                variances.append(Z.var().values)
                imputed_datasets.append(Z.copy())

            self.imputed_datasets_ = imputed_datasets
            self.estimates_ = estimates
            self.variances_ = variances
            logging.info("MIMCA model fitting completed successfully.")
            return self
        except Exception as e:
            logging.error(f"Error fitting MIMCA model: {e}")
            raise

    def transform(self):
        """Transform the data using the fitted MIMCA model."""
        try:
            return self.imputed_datasets_
        except Exception as e:
            logging.error(f"Error transforming data: {e}")
            raise

def convert_to_categorical(Z, original_columns):
    """Convert disjunctive table back to categorical data."""
    try:
        categorical_data = pd.DataFrame(index=Z.index)
        for col in original_columns:
            subframe = Z[[c for c in Z.columns if c.startswith(col)]]
            subframe = subframe.div(subframe.sum(axis=1), axis=0).fillna(1 / len(subframe.columns))
            categorical_data[col] = subframe.apply(lambda row: np.random.choice(subframe.columns, p=row), axis=1)
            categorical_data[col] = categorical_data[col].str.replace(f'^{col}_', '', regex=True)
        return categorical_data
    except Exception as e:
        logging.error(f"Error converting to categorical data: {e}")
        raise

