import pickle
import inspect

import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
import numpy as np
import scipy.stats

from scalib import ScalibError
from scalib.modeling import LDAClassifier, MultiLDA, Lda, LdaAcc


def get_rng(**args):
    arg = tuple(args.items())
    # Hash caller name (i.e. test name) to get the rng seed.
    # arg is a supplementary hash identifier
    return np.random.default_rng(seed=abs(hash((arg, inspect.stack()[1][3]))))


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
    # np.set_printoptions(threshold=np.inf)
    ns = 10
    n_components = 2
    nc = 4
    n = 5000

    rng = get_rng()
    m = rng.integers(0, 100, (nc, ns))
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components)
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

    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, n, dtype=np.uint16)
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
    # np.set_printoptions(threshold=np.inf)
    ns = 10
    n_components = 2
    nc = 4
    n = 500

    rng = get_rng()
    m = rng.integers(0, 100, (nc, ns))
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components)
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

    # Verify value of means by accessor
    ref_means = lda_ref.means_
    lda_means = lda.get_mus()
    assert np.allclose(ref_means, lda_means, rtol=1e-10)

    # Verify scatter matrix by accessor
    lda_scat = lda.get_sw() / n
    cov_ref = lda_ref.covariance_
    assert np.allclose(lda_scat, cov_ref, rtol=1e-5)

    # Not point of comparison with sklearn, but we here call
    # the accessor for the inter-class scatter matrix just to verify
    # that its worling.
    smat = lda.get_sb()

    # We can't do much more since sklearn has no way to reduce dimensionality of LDA.
    # e.g., comparing probas will fail


def test_lda_noproj():
    ns = 10
    n_components = 3
    nc = 4
    n = 500

    rng = get_rng()
    m = rng.integers(0, 100, (nc, ns))
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    lda = LDAClassifier(nc, n_components)
    lda.fit_u(traces, labels, 1)
    lda.solve()

    lda_ref = LDA_sklearn(solver="eigen", n_components=n_components)
    lda_ref.fit(traces, labels)

    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    prs = lda.predict_proba(traces)
    prs_ref = lda_ref.predict_proba(traces)

    assert np.allclose(prs, prs_ref, rtol=1e-2, atol=1e-3)


def test_lda_fail_bad_traces():
    # Issue #56
    n = 100
    ns = 6
    nc = 4
    lda = LDAClassifier(nc, 3)
    rng = get_rng()
    traces_bad = rng.integers(0, 1, (n, ns), dtype=np.int16)
    y = rng.integers(0, nc, n, dtype=np.uint16)
    lda.fit_u(traces_bad, y, 0)
    with pytest.raises(ScalibError):
        lda.solve()


def test_multilda():
    rng = get_rng()
    x = rng.integers(0, 256, (5000, 50), dtype=np.int16)
    y = rng.integers(0, 256, (5000, 5), dtype=np.uint16)
    pois = [list(range(7 * i, 7 * i + 10)) for i in range(5)]
    lda = MultiLDA(5 * [256], 5 * [3], pois)
    lda.fit_u(x, y)
    lda.solve()
    x = rng.integers(0, 256, (20, 50), dtype=np.int16)
    _ = lda.predict_proba(x)


def multi_lda_data_indep(ns, nc, nv, n, n_batches, rng):
    y = [rng.integers(0, nc, (n, nv), dtype=np.uint16) for _ in n_batches]
    traces = [
        rng.integers(-(2**15), 2**15, (n, ns), dtype=np.uint16) for _ in n_batches
    ]
    return traces, y


def multi_lda_gen_pois_overlap(rng, ns, nv, npois):
    pois = np.tile(np.arange(ns), (nv, 1))
    rng.shuffle(pois, axis=1)
    return pois[:, :npois]


def multi_lda_gen_pois_consec(nv, npois, gap=0):
    return np.array(
        [np.arange(i * (npois + gap), i * (npois + gap) + npois) for i in range(nv)]
    )


def multi_lda_gen_indep_overlap(rng, ns, nc, nv, npois, n, n_batches, maxl=2**15, **_):
    pois = np.tile(np.arange(ns), (nv, 1))
    rng.shuffle(pois, axis=1)
    pois = pois[:, :npois]
    y = [rng.integers(0, nc, (n, nv), dtype=np.uint16) for _ in range(n_batches)]
    traces = [
        rng.integers(-maxl, maxl, (n, ns), dtype=np.int16) for _ in range(n_batches)
    ]
    return pois, traces, y


def multi_lda_compare(nc, nv, p, pois, traces, x, **_):
    ncs = [nc for _ in range(nv)]
    ps = [p for _ in range(nv)]
    multi_lda = MultiLDA(ncs, ps, pois=pois)
    multi_lda3 = LdaAcc(pois=pois, nc=nc)
    for t, y in zip(traces, x):
        multi_lda.fit_u(t, y)
        multi_lda3.fit_u(t, y)
    for mus, mus3 in zip(multi_lda.get_mus(), multi_lda3.get_mus()):
        assert np.allclose(mus, mus3)
    for sb, sb3 in zip(multi_lda.get_sb(), multi_lda3.get_sb()):
        assert np.allclose(sb, sb3)
    for sw, sw3 in zip(multi_lda.get_sw(), multi_lda3.get_sw()):
        assert np.allclose(sw, sw3)
    multi_lda = multi_lda3._ldas(p)
    multi_lda3 = Lda(multi_lda3, p=p)
    for t in traces:
        probas = multi_lda.predict_proba(t)
        probas3 = multi_lda3.predict_proba(t)
        assert np.allclose(probas, probas3)


def test_multi_lda_compare():
    cases = [
        dict(ns=2, nc=2, nv=1, npois=2, n=10, n_batches=1, p=1),
        dict(ns=2, nc=2, nv=1, npois=2, n=5, n_batches=2, p=1),
        dict(ns=3, nc=2, nv=1, npois=2, n=20, n_batches=1, p=1, maxl=100),
        dict(ns=20, nc=4, nv=4, npois=5, n=10, n_batches=3, p=2),
        dict(ns=100, nc=256, nv=2, npois=20, n=500, n_batches=5, p=4),
        dict(ns=1000, nc=4, nv=10, npois=2, n=10, n_batches=5, p=1),
    ]
    for case in cases:
        rng = get_rng(**case)
        print(80 * "#" + "\ncase:", case)
        pois, traces, x = multi_lda_gen_indep_overlap(rng, **case)
        multi_lda_compare(pois=pois, traces=traces, x=x, **case)


def test_simple_multi_lda_compare():
    nc = 2
    nv = 1
    pois = [np.array([0, 1])]
    p = 1
    traces = [np.array([[1, 0], [0, 1], [0, 0], [4, 4]], dtype=np.int16)]
    x = [np.array([[0], [0], [1], [1]]).astype(np.uint16)]
    lda = LDAClassifier(nc, p)
    lda.fit_u(traces[0], x[0][:, 0])
    lda.solve()
    multi_lda_compare(nc=nc, nv=nv, p=p, pois=pois, traces=traces, x=x)


def test_seq_multi_lda_compare():
    nv = 100
    npois = 5
    nc = 4
    rng = np.random.default_rng(seed=0)
    _, traces, x = multi_lda_gen_indep_overlap(
        rng, ns=nv * npois, nc=nc, nv=nv, npois=0, n=1000, n_batches=1
    )
    pois = [list(range(i * npois, (i + 1) * npois)) for i in range(nv)]
    multi_lda_compare(nc=nc, nv=nv, p=2, pois=pois, traces=traces, x=x)


def test_multi_lda_pickle():
    ns = 10
    nc = 4
    n = 5000
    rng = get_rng()
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, (n, 1), dtype=np.uint16)
    lda_acc = LdaAcc(pois=[list(range(ns))], nc=nc)
    lda_acc.fit_u(traces, labels)
    dumped_lda_acc = pickle.dumps(lda_acc)
    lda_acc2 = pickle.loads(dumped_lda_acc)
    lda = Lda(lda_acc, p=2)
    lda2 = Lda(lda_acc2, p=2)

    dumped_lda = pickle.dumps(lda2)
    lda2 = pickle.loads(dumped_lda)

    prs = lda.predict_proba(traces)
    prs2 = lda2.predict_proba(traces)

    assert np.allclose(prs, prs2)


def multi_lda_select_simple(rng, nv, ns, npois, nv_sel, n_sel, maxl=2**15):
    nc = 4
    n = 100
    traces = rng.integers(-maxl, maxl, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, (n, nv), dtype=np.uint16)
    pois = list(np.random.permutation(range(ns))[:npois] for _ in range(nv))
    lda_acc = LdaAcc(pois=pois, nc=nc)
    lda_acc.fit_u(traces, labels)
    lda_all = Lda(lda_acc, p=1)
    prs_all = lda_all.predict_proba(traces)
    # Validate for a random selection
    for _ in range(n_sel):
        selection = list(np.random.permutation(range(nv))[:nv_sel])
        lda_s_only = lda_all.select_variables(selection)
        prt = lda_s_only.predict_proba(traces)
        assert np.allclose(prs_all[selection, ...], prt)


def test_multi_lda_select():
    cases = [
        dict(nv=5, ns=25, npois=1, nv_sel=5, n_sel=2),
        dict(nv=5, ns=25, npois=1, nv_sel=1, n_sel=2),
        dict(nv=5, ns=25, npois=1, nv_sel=2, n_sel=2),
        dict(nv=5, ns=25, npois=5, nv_sel=2, n_sel=2),
        dict(nv=5, ns=25, npois=15, nv_sel=2, n_sel=1),
    ]
    for case in cases:
        rng = get_rng(**case)
        multi_lda_select_simple(rng, **case)
