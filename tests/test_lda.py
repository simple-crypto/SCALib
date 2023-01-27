import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
from scalib.modeling import LDAClassifier
import numpy as np
import scipy.stats
import pickle


def is_parallel(x, y):
    z = np.abs(x.dot(y))
    z2 = np.linalg.norm(x) * np.linalg.norm(y)
    return np.allclose(z, z2, rtol=1e-3)


def parallel_factor(x, y):
    """Return 1 if x and y are parallel and in the same direction, and -1 is
    they are in opposite directions.
    """
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.dot(x, y)


def test_lda_pickle():
    np.set_printoptions(threshold=np.inf)  # for debug
    ns = 10
    n_components = 2
    nc = 4
    n = 5000

    m = np.random.randint(0, 100, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components, ns)
    lda.fit_u(traces, labels, 1)
    lda.solve()

    dumped_lda = pickle.dumps(lda)
    lda = pickle.loads(dumped_lda)

    lda_ref = LDA_sklearn(solver="eigen", n_components=n_components)
    lda_ref.fit(traces, labels)

    lda_projection = lda.lda.get_projection()
    lda_ref_projection = lda_ref.scalings_[:, :n_components]
    projections_similar = all(
        is_parallel(x, y) for x, y in zip(lda_projection.T, lda_ref_projection.T)
    )
    print("lda_projection")
    print(lda_projection)
    print("lda_ref_projection")
    print(lda_ref_projection)
    assert projections_similar, (lda_projection, lda_ref_projection)
    # To correct if projection vectors are opposite.
    parallel_factors = np.array(
        [parallel_factor(x, y) for x, y in zip(lda_projection.T, lda_ref_projection.T)]
    )

    print("parallel_factors")
    print(parallel_factors)

    # generate means and cov in subspace
    traces_t = (lda_ref.scalings_[:, :n_components].T @ traces.T).T
    means_check = np.zeros((nc, n_components))
    for i in range(nc):
        I = np.where(labels == i)[0]
        means_check[i, :] = np.mean(traces_t[I, :], axis=0)
    traces_t = traces_t - means_check[labels, :]
    cov_check = np.cov(traces_t.T)

    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    prs = lda.predict_proba(traces)
    traces_t = (lda_ref.scalings_[:, :n_components].T @ traces.T).T
    prs_ref = np.zeros((len(traces), nc))
    for x in range(nc):
        prs_ref[:, x] = scipy.stats.multivariate_normal.pdf(
            traces_t, means_check[x], cov_check
        )
    prs_ref = (prs_ref.T / np.sum(prs_ref, axis=1)).T

    assert np.allclose(prs, prs_ref, rtol=1e-2)


def test_lda():
    np.set_printoptions(threshold=np.inf)  # for debug
    ns = 10
    n_components = 2
    nc = 4
    n = 500

    m = np.random.randint(0, 100, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components, ns)
    lda.fit_u(traces, labels, 1)
    lda.solve()

    lda_ref = LDA_sklearn(solver="eigen")
    lda_ref.fit(traces, labels)

    lda_projection = lda.lda.get_projection()
    lda_ref_projection = lda_ref.scalings_[:, :n_components]
    projections_similar = all(
        is_parallel(x, y) for x, y in zip(lda_projection.T, lda_ref_projection.T)
    )
    print("lda_projection")
    print(lda_projection)
    print("lda_ref_projection")
    print(lda_ref_projection)
    assert projections_similar, (lda_projection, lda_ref_projection)
    # We can't do much more since sklearn has no way to reduce dimensionality of LDA.
    # e.g., comparing probas will fail


def test_lda_noproj():
    ns = 10
    n_components = 3
    nc = 4
    n = 500

    m = np.random.randint(0, 100, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components, ns)
    lda.fit_u(traces, labels, 1)
    lda.solve()

    lda_ref = LDA_sklearn(solver="eigen", n_components=n_components)
    lda_ref.fit(traces, labels)

    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    prs = lda.predict_proba(traces)
    prs_ref = lda_ref.predict_proba(traces)

    assert np.allclose(prs, prs_ref, rtol=1e-2, atol=1e-3)
