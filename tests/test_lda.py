import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
from scalib.modeling import LDAClassifier
import numpy as np
import scipy.stats


def test_lda():
    np.set_printoptions(threshold=np.inf)  # for debug
    ns = 10
    n_components = 2
    nc = 4
    n = 5000

    m = np.random.randint(0, 4, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components, ns)
    lda.fit(traces, labels)

    lda_ref = LDA_sklearn(solver="eigen")
    lda_ref.fit(traces, labels)

    assert np.allclose(lda.lda.get_projection(), lda_ref.scalings_[:, :n_components])

    # check equivalence for means and cov in subspace
    traces_t = (lda_ref.scalings_[:, :n_components].T @ traces.T).T
    means_check = np.zeros((nc, n_components))
    for i in range(nc):
        I = np.where(labels == i)[0]
        means_check[i, :] = np.mean(traces_t[I, :], axis=0)
    traces_t = traces_t - means_check[labels, :]
    cov_check = np.cov(traces_t.T)
    assert np.allclose(means_check, lda.lda.get_means().T)
    #    assert(np.allclose(cov_check,lda.lda.get_cov()))

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
