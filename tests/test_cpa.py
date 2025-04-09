import pytest
import numpy as np
import hashlib
import inspect

from scalib.attacks import Cpa


def get_rng(**args):
    """
    Hash caller name (i.e. test name) to get the rng seed.
    args are also hashed in the seed.
    """
    # Use a deterministic hash (no need for cryptographic robustness, but
    # python's hash() is randomly salted).
    seed = hashlib.sha256((repr(args) + inspect.stack()[1][3]).encode()).digest()
    return np.random.Generator(np.random.PCG64(list(seed)))


def pearson_corr(x, y):
    xc = x - np.mean(x, axis=0)
    yc = y - np.mean(y, axis=0)
    cov = np.mean(xc * yc, axis=0)

    x_std = np.std(x, axis=0)
    y_std = np.std(y, axis=0)

    return cov / (x_std * y_std)


def pearson_corr_refv1(x, labels, model):
    """
    x: (n, ns)
    labels (n, nv)
    model: (nv, nc, ns)

    Returns:
    the correlation of shape (nv, ns)
    """
    # fetch config
    n = x.shape[0]
    nv = model.shape[0]
    ns = x.shape[1]
    # Compute the centered data
    xc = x - np.mean(x, axis=0)
    # Compute the centered model
    mus = np.mean(model, axis=1)
    mc = model.copy()
    for i in range(nv):
        mc[i] -= mus[i]
    # Compute the data std
    xstd = np.std(x, axis=0, ddof=1)
    # Compute the model std (exact variance here)
    v_mstd = np.std(model, axis=1, ddof=0)
    # Compute the unbiased covariance estimation
    u_mc = np.zeros([nv, n, ns])
    for ni in range(nv):
        # Create centered model corresponding to variable
        for i, lab in enumerate(labels[:, ni]):
            u_mc[ni, i, :] = mc[ni, lab, :]
        # multiply with the centered data
        u_mc[ni, :, :] *= xc
    v_cov = np.sum(u_mc, axis=1) / (n - 1)
    # divide by the data std
    for vi in range(nv):
        v_cov[vi, :] /= xstd
    rcov = v_cov / v_mstd
    return rcov


def cpa_inner_test(seed, ns, nc, n, nv):
    rng = get_rng()
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, (n, nv), dtype=np.uint16)
    models = rng.random((nv, nc, ns), dtype=np.float64)

    ### Create the CPA and fit all with SCAlib
    cpa = Cpa(nc, Cpa.Xor)
    cpa.fit_u(traces, labels)
    corr = cpa.get_correlation(
        models,
    )

    ### Get the reference now
    corr_ref = pearson_corr_refv1(traces, labels, models)

    for i, (cv, cvr) in enumerate(zip(corr[:, 0, :], corr_ref)):
        assert np.allclose(
            cv, cvr
        ), "[INNER-seed:{}-ns:{}-nc:{}-n:{}-nv:{}]\ncorr\n{}\nref\n{}".format(
            seed, ns, nc, n, nv, cv, cvr
        )


def test_cpa_full():
    cpa_inner_test(0, 1, 2, 10, 1)
    cpa_inner_test(0, 1000, 2, 10, 1)
    cpa_inner_test(0, 1, 256, 1000, 1)
    cpa_inner_test(0, 1000, 256, 1000, 1)
    cpa_inner_test(0, 1, 2, 10, 2)
    cpa_inner_test(0, 1000, 256, 1000, 5)
