import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import ClassicalMDS
from sklearn.metrics import euclidean_distances


def test_classical_mds_equivalent_to_pca():
    X, _ = load_iris(return_X_y=True)

    cmds = ClassicalMDS(n_components=2, metric="euclidean")
    pca = PCA(n_components=2)

    Z1 = cmds.fit_transform(X)
    Z2 = pca.fit_transform(X)

    # Swap the signs if necessary
    for comp in range(2):
        if Z1[0, comp] < 0 and Z2[0, comp] > 0:
            Z2[:, comp] *= -1

    assert_allclose(Z1, Z2)

    assert_allclose(np.sqrt(cmds.eigenvalues_), pca.singular_values_)


def test_classical_mds_equivalent_on_data_and_distances():
    X, _ = load_iris(return_X_y=True)

    cmds = ClassicalMDS(n_components=2, metric="euclidean")
    Z1 = cmds.fit_transform(X)

    cmds = ClassicalMDS(n_components=2, metric="precomputed")
    Z2 = cmds.fit_transform(euclidean_distances(X))

    assert_allclose(Z1, Z2)


def test_classical_mds_wrong_inputs():
    # Non-symmetric input
    dissim = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    with pytest.raises(ValueError, match="Array must be symmetric"):
        ClassicalMDS(metric="precomputed").fit(dissim)

    # Non-square input
    dissim = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="array must be 2-dimensional and square"):
        ClassicalMDS(metric="precomputed").fit(dissim)


def test_classical_mds_metric_params():
    X, _ = load_iris(return_X_y=True)

    cmds = ClassicalMDS(n_components=2, metric="euclidean")
    Z1 = cmds.fit_transform(X)

    cmds = ClassicalMDS(n_components=2, metric="minkowski", metric_params={"p": 2})
    Z2 = cmds.fit_transform(X)

    assert_allclose(Z1, Z2)

    cmds = ClassicalMDS(n_components=2, metric="minkowski", metric_params={"p": 1})
    Z3 = cmds.fit_transform(X)

    assert not np.allclose(Z1, Z3)


def test_classical_mds_non_euclidean_no_nan():
    # Non-Euclidean precomputed dissimilarities (the defining PCoA/Torgerson
    # use case) produce negative eigenvalues of the double-centered matrix.
    # Those dimensions cannot be embedded in Euclidean space and must be
    # clipped to zero; otherwise np.sqrt of a negative eigenvalue silently
    # fills the embedding with NaNs. Non-regression test for a bug where
    # negative eigenvalues were passed to np.sqrt unclipped.
    rng = np.random.RandomState(12)
    A = rng.rand(5, 5)
    dissimilarity = np.abs(A + A.T)
    np.fill_diagonal(dissimilarity, 0)

    est = ClassicalMDS(n_components=4, metric="precomputed").fit(dissimilarity)

    assert np.isfinite(est.embedding_).all()
    assert np.all(est.eigenvalues_ >= 0)


def test_classical_mds_rank_deficient_no_nan():
    # When n_components exceeds the numerical rank of Euclidean input, the
    # excess eigenvalues returned by eigh are tiny and frequently slightly
    # negative due to floating-point error, which np.sqrt turns into NaNs.
    # Here the five points are collinear (rank 1) while n_components=4.
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    est = ClassicalMDS(n_components=4, metric="euclidean").fit(X)

    assert np.isfinite(est.embedding_).all()
    assert np.all(est.eigenvalues_ >= 0)
