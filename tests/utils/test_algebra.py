import numpy as np
from sympy import diag

from qolmat.utils import algebra


def test_frechet_distance_exact():
    means1 = np.array([0, 1, 3])
    stds = np.array([1, 1, 1])
    cov1 = np.diag(stds**2)

    means2 = np.array([0, -1, 1])
    cov2 = np.eye(3, 3)

    expected = np.sum((means2 - means1) ** 2) + np.sum((np.sqrt(stds) - 1) ** 2)
    expected /= 3
    result = algebra.frechet_distance_exact(means1, cov1, means2, cov2)
    np.testing.assert_almost_equal(result, expected, decimal=3)


def test_kl_divergence_gaussian_exact():
    means1 = np.array([0, 1, 3])
    stds = np.array([1, 2, 3])
    cov1 = np.diag(stds**2)

    means2 = np.array([0, -1, 1])
    cov2 = np.eye(3, 3)

    expected = (np.sum(stds**2 - np.log(stds**2) - 1 + (means2 - means1) ** 2)) / 2
    result = algebra.kl_divergence_gaussian_exact(means1, cov1, means2, cov2)
    np.testing.assert_almost_equal(result, expected, decimal=3)
