import numpy as np

from qolmat.utils import algebra
from qolmat.utils.algebra import svdtriplet


def test_frechet_distance_exact():
    means1 = np.array([0, 1, 3])
    stds = np.array([1, 1, 1])
    cov1 = np.diag(stds**2)

    means2 = np.array([0, -1, 1])
    cov2 = np.eye(3, 3)

    expected = np.sum((means2 - means1) ** 2) + np.sum(
        (np.sqrt(stds) - 1) ** 2
    )
    expected /= 3
    result = algebra.frechet_distance_exact(means1, cov1, means2, cov2)
    np.testing.assert_almost_equal(result, expected, decimal=3)


def test_kl_divergence_gaussian_exact():
    means1 = np.array([0, 1, 3])
    stds = np.array([1, 2, 3])
    cov1 = np.diag(stds**2)

    means2 = np.array([0, -1, 1])
    cov2 = np.eye(3, 3)

    expected = (
        np.sum(stds**2 - np.log(stds**2) - 1 + (means2 - means1) ** 2)
    ) / 2
    result = algebra.kl_divergence_gaussian_exact(means1, cov1, means2, cov2)
    np.testing.assert_almost_equal(result, expected, decimal=3)

def test_svdtriplet_known_matrix():
    """Test svdtriplet on a known matrix without weights."""
    X = np.array([[3, 1], [1, 3]])
    expected_singular_values = np.array([4, 2])
    expected_U = np.array([[0.7071, -0.7071],
                           [0.7071, 0.7071]])
    expected_V = np.array([[0.7071, 0.7071],
                           [0.7071, -0.7071]])
    # Call svdtriplet without weights
    s, U, V = svdtriplet(X, row_w=None, ncp=2)
    # Compare singular values
    np.testing.assert_almost_equal(s, expected_singular_values, decimal=3)
    np.testing.assert_almost_equal(np.abs(U), np.abs(expected_U), decimal=3)
    np.testing.assert_almost_equal(np.abs(V), np.abs(expected_V), decimal=3)

def test_svdtriplet_with_row_weights():
    """Test svdtriplet with row weights."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    row_w = np.array([0.2, 0.5, 0.3])
    # Manually compute the weighted X
    X_weighted = X * np.sqrt(row_w)[:, None]
    U_expected, s_expected, Vt_expected = np.linalg.svd(X_weighted,
                                                        full_matrices=False)
    V_expected = Vt_expected.T
    # Call svdtriplet with weights
    s, U, V = svdtriplet(X, row_w=row_w, ncp=2)
    # Rescale U_expected by dividing by sqrt(row_w)
    U_expected /= np.sqrt(row_w)[:, None]
    # Compare singular values
    np.testing.assert_allclose(s, s_expected[:2], atol=1e-6)
    # Compare U and V (up to sign)
    np.testing.assert_allclose(np.abs(U), np.abs(U_expected[:, :2]), atol=1e-6)
    np.testing.assert_allclose(np.abs(V), np.abs(V_expected[:, :2]), atol=1e-6)

def test_svdtriplet_ncp_limit():
    """Test svdtriplet with ncp less than the full rank."""
    X = np.random.rand(5, 3)
    ncp = 2
    s, U, V = svdtriplet(X, ncp=ncp)
    # Check the dimensions
    assert s.shape == (ncp,)
    assert U.shape == (X.shape[0], ncp)
    assert V.shape == (X.shape[1], ncp)
    # Reconstruct X approximation
    X_approx = U @ np.diag(s) @ V.T
    # Check that the approximation is close to X
    # Note: With reduced ncp, approximation won't be exact
    assert X_approx.shape == X.shape
    s_full, _, _ = svdtriplet(X)
    X_full = U @ np.diag(s_full) @ V.T
    error_ncp = np.linalg.norm(X - X_approx)
    error_full = np.linalg.norm(X - X_full)
    assert error_ncp >= error_full

def test_svdtriplet_row_weights_none():
    """Test svdtriplet with default row weights."""
    X = np.random.rand(4, 4)
    s_default, U_default, V_default = svdtriplet(X)
    # Manually set uniform weights
    row_w = np.ones(X.shape[0]) / X.shape[0]
    s_manual, U_manual, V_manual = svdtriplet(X, row_w=row_w)
    # Compare results
    np.testing.assert_allclose(s_default, s_manual, atol=1e-6)
    np.testing.assert_allclose(U_default, U_manual, atol=1e-6)
    np.testing.assert_allclose(V_default, V_manual, atol=1e-6)

def test_svdtriplet_zero_matrix():
    """Test svdtriplet on a zero matrix."""
    X = np.zeros((3, 3))
    s, U, V = svdtriplet(X)
    # Singular values should be zero
    expected_s = np.zeros(3)
    np.testing.assert_array_equal(s, expected_s)
    # U and V should be orthogonal matrices
    np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-6)

def test_svdtriplet_non_square_matrix():
    """Test svdtriplet on a non-square matrix."""
    X = np.random.rand(6, 4)
    s, U, V = svdtriplet(X)
    # Check dimensions
    assert U.shape == (6, 4)
    assert s.shape == (4,)
    assert V.shape == (4, 4)
    # Reconstruct X
    X_reconstructed = U @ np.diag(s) @ V.T
    np.testing.assert_allclose(X, X_reconstructed, atol=1e-6)

def test_svdtriplet_large_ncp():
    """Test svdtriplet with ncp larger than possible."""
    X = np.random.rand(5, 3)
    ncp = 10  # Larger than min(n_samples - 1, n_features)
    s, U, V = svdtriplet(X, ncp=ncp)
    expected_ncp = min(5 - 1, 3)
    assert s.shape == (expected_ncp,)
    assert U.shape == (5, expected_ncp)
    assert V.shape == (3, expected_ncp)

def test_svdtriplet_negative_weights():
    """Test svdtriplet with negative row weights (should raise an error)."""
    X = np.random.rand(4, 4)
    row_w = np.array([0.25, -0.25, 0.5, 0.5])  # Negative weight
    with pytest.raises(ValueError):
        s, U, V = svdtriplet(X, row_w=row_w)




