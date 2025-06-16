import numpy as np
from scipy.linalg import eigh
from scalib.modeling import RLDAClassifier
from scalib.metrics import RLDAInformationEstimator
import pytest
from scalib import ScalibError


def label2bits(label, nbits):
    """Return the array of -1/1 coefficients corresponding to a label."""
    return [1] + [-1 if (label >> i) & 1 == 0 else 1 for i in range(nbits)]


def get_bits(data, nbits):
    assert data.ndim == 1
    return np.array([label2bits(int(x), nbits) for x in data], dtype=np.int8)


def rlda_inner_test(ns, n_components, nb, nv):
    nc = 2**nb
    n_model = 2000
    n_test = 500
    noise = 2000
    assert nv <= ns, "Test case error, use higher ns"
    # Generate profiling data
    m = np.ones((nv, nc, ns), dtype=np.int16)
    for v in range(nv):
        m[v, :, v] = (get_bits(np.arange(nc), nbits=nb).astype(np.int64) + 1).sum(
            axis=1
        ) * (v + 1)
    traces = np.random.randint(-noise, noise, (n_model, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, (nv, n_model), dtype=np.uint64)
    for v in range(nv):
        traces += m[v, labels[v]].astype(np.int16)
    # Generate test data
    test_traces = np.random.randint(-noise, noise, (n_test, ns), dtype=np.int16)
    test_labels = np.random.randint(0, nc, (nv, n_test), dtype=np.uint16)
    for v in range(nv):
        test_traces += m[v, test_labels[v]]

    ## Run SCALib RLDA and compare projection matrices
    rlda = RLDAClassifier(nb, n_components)
    rlda.fit_u(traces, labels.T, 1)
    rlda.solve()

    for v in range(nv):
        ## Start computing RLDA in python
        label_bits = get_bits(labels[v], nbits=nb).astype(np.int64)

        B = label_bits.T @ label_bits
        C = label_bits.T @ traces
        S_L = traces.astype(np.int64).T @ traces.astype(np.int64)

        A = np.linalg.solve(B, C)
        mu = C[0] / n_model
        SB = A.T @ B @ A - n_model * np.outer(mu, mu)
        SW = S_L + A.T @ B @ A - C.T @ A - A.T @ C

        ev, W = eigh(SB, SW, subset_by_index=(ns - n_components, ns - 1))
        W = W.T
        cov = W @ SW @ W.T * 1 / n_model

        evals, evecs = np.linalg.eigh(cov)
        Wnorm = evecs * (evals[:, np.newaxis] ** -0.5)

        Weff = W.T @ Wnorm
        Aeff = A @ Weff

        prs_ref = (test_traces @ Weff)[:, :, np.newaxis] - (
            get_bits(np.arange(2**nb, dtype=np.uint64), nbits=nb) @ Aeff
        ).T[np.newaxis]
        prs_ref = np.exp(-0.5 * (prs_ref**2).sum(axis=1))
        prs_ref = prs_ref / prs_ref.sum(axis=1, keepdims=True)

        prs = rlda.predict_proba(test_traces, v)

        assert np.allclose(prs, prs_ref)


def test_rlda():
    ns = 6
    n_components = 3
    for nv in [1, 2, 4]:
        for nb in [1, 4, 8, 12]:
            rlda_inner_test(ns, min(nb, n_components), nb, nv)


def test_information():
    # test clustering + information
    ns = 6
    n_components = 3
    nb = 8
    nc = 2**nb
    n_model = 2000
    n_test = 2000
    noise = 30
    # Generate profiling data
    m = np.ones((nc, ns), dtype=np.int16)
    m[:, 0] = np.arange(nc)
    traces = np.random.randint(-noise, noise, (n_model, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, (n_model,), dtype=np.uint64)
    traces += m[labels].astype(np.int16)
    # Generate test data
    test_traces = np.random.randint(-noise, noise, (n_test, ns), dtype=np.int16)
    test_labels = np.random.randint(0, nc, n_test, dtype=np.uint16)
    test_traces += m[test_labels]

    rlda = RLDAClassifier(nb, n_components)
    rlda.fit_u(traces, labels[:, np.newaxis], 1)
    rlda.solve()
    prs = rlda.predict_proba(test_traces, 0)

    cl = rlda.get_clustered_model(0, 0, 2**16, False)
    it = RLDAInformationEstimator(cl, 0)
    # pi bounds should be exact bounds as t=0
    it.fit_u(test_traces, test_labels.astype(np.uint64))
    pi = it.get_information()
    pi_ref = nb + np.log2(prs[np.arange(n_test), test_labels]).mean()
    assert np.allclose(pi, pi_ref)


def test_rlda_fail_empty_classes():
    ns = 6
    n_components = 3
    nb = 8
    nc = 2**nb
    n_model = 2000
    noise = 30
    # Generate profiling data
    m = np.ones((nc, ns), dtype=np.int16)
    m[:, 0] = np.arange(nc)
    traces = np.random.randint(-noise, noise, (n_model, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, (n_model,), dtype=np.uint64)
    traces += m[labels].astype(np.int16)

    # make sure some bits have always the same value, here zero
    labels &= (nc >> 2) - 1

    rlda = RLDAClassifier(nb, n_components)
    rlda.fit_u(traces, labels[:, np.newaxis], 1)

    with pytest.raises(ScalibError):
        rlda.solve()


def generate_rlda_inputs(seed, ns, n, nv, nb, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = rng
    traces = rng.integers(0, 100, size=(n, ns), dtype=np.int16)
    labels = rng.integers(0, 2**nb, size=(n, nv), dtype=np.uint64)
    return rng, dict(traces=traces, labels=labels)


def subtest_rlda_pred_log2p1(seed, ns, nb, n, nv, p):
    # Generate the inputs
    rng, data = generate_rlda_inputs(seed, ns, n, nv, nb)
    # RLDA
    rlda = RLDAClassifier(nb, p)
    rlda.fit_u(data["traces"], data["labels"], 1)
    rlda.solve()

    # new traces
    nntrs = 10
    rng, ndata = generate_rlda_inputs(seed + 1, ns, nntrs, nv, nb, rng=rng)

    l2p1s = rlda.predict_log2p1(ndata["traces"], ndata["labels"])

    # Verify all probas
    for nvi in range(nv):
        lprs = np.log2(rlda.predict_proba(ndata["traces"], nvi))
        cpick = lprs[np.arange(nntrs), ndata["labels"][:, nvi]]
        assert np.allclose(l2p1s[nvi, :], cpick), f"{seed};{ns};{nb};{n};{nv};{p};{nvi}"


def test_rlda_pred_log2p1():
    # seed, ns, nb, n, nv, p
    subtest_rlda_pred_log2p1(0, 2, 2, 16, 1, 1)
    subtest_rlda_pred_log2p1(0, 2, 2, 16, 5, 1)
    subtest_rlda_pred_log2p1(0, 3, 4, 64, 2, 1)
    subtest_rlda_pred_log2p1(0, 3, 4, 64, 5, 1)
    subtest_rlda_pred_log2p1(0, 3, 4, 64, 5, 2)
    subtest_rlda_pred_log2p1(0, 3, 9, 1000, 5, 2)
