import numpy as np
import scipy
from numpy.typing import NDArray, ArrayLike


def frechet_distance_exact(
    means1: NDArray,
    cov1: NDArray,
    means2: NDArray,
    cov2: NDArray,
) -> float:
    """Compute the Fréchet distance between two dataframes df1 and df2
    Frechet_distance = || mu_1 - mu_2 ||_2^2 + Tr(Sigma_1 + Sigma_2 - 2(Sigma_1 . Sigma_2)^(1/2))
    It is normalized, df1 and df2 are first scaled by a factor (std(df1) + std(df2)) / 2
    and then centered around (mean(df1) + mean(df2)) / 2
    The result is divided by the number of samples to get an homogeneous result.
    Based on: Dowson, D. C., and BV666017 Landau. "The Fréchet distance between multivariate normal
    distributions." Journal of multivariate analysis 12.3 (1982): 450-455.

    Parameters
    ----------
    means1 : NDArray
        Means of the first distribution
    cov1 : NDArray
        Covariance matrix of the first distribution
    means2 : NDArray
        Means of the second distribution
    cov2 : NDArray
        Covariance matrix of the second distribution

    Returns
    -------
    float
        Frechet distance
    """
    n = len(means1)
    if (means2.shape != (n,)) or (cov1.shape != (n, n)) or (cov2.shape != (n, n)):
        raise ValueError("Inputs have to be of same dimensions.")

    ssdiff = np.sum((means1 - means2) ** 2.0)
    product = np.array(cov1 @ cov2)
    if product.ndim < 2:
        product = product.reshape(-1, 1)
    covmean = scipy.linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    frechet_dist = ssdiff + np.trace(cov1 + cov2 - 2.0 * covmean)

    return frechet_dist / n


def kl_divergence_gaussian_exact(
    means1: NDArray, cov1: NDArray, means2: NDArray, cov2: NDArray
) -> float:
    """
    Exact Kullback-Leibler divergence computed between two multivariate normal distributions
    Based on https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Parameters
    ----------
    means1: NDArray
        Mean of the first distribution
    cov1: NDArray
        Covariance matrx of the first distribution
    means2: NDArray
        Mean of the second distribution
    cov2: NDArray
        Covariance matrx of the second distribution
    Returns
    -------
    float
        Kulback-Leibler divergence
    """
    n_variables = len(means1)
    L1, _ = scipy.linalg.cho_factor(cov1)
    L2, _ = scipy.linalg.cho_factor(cov2)
    M = scipy.linalg.solve(L2, L1)
    y = scipy.linalg.solve(L2, means2 - means1)
    norm_M = (M**2).sum().sum()
    norm_y = (y**2).sum()
    term_diag_L = 2 * np.sum(np.log(np.diagonal(L2) / np.diagonal(L1)))
    div_kl = 0.5 * (norm_M - n_variables + norm_y + term_diag_L)
    return div_kl
