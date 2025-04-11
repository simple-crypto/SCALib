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


def cpa_inner_intermediate(seed, ns, nc, n, nv, perm_internal, intermediate):
    rng = get_rng()
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, (n, nv), dtype=np.uint16)
    models = rng.random((nv, nc, ns), dtype=np.float64)

    ### Create the CPA and fit all with SCAlib
    cpa = Cpa(nc, intermediate)
    cpa.fit_u(traces, labels)
    corr = cpa.get_correlation(
        models,
    )

    ### Get the reference, corresponding to the intermediate with k = 0
    ### Verify validity for every class
    for c in range(nc):
        models_used = models[:, perm_internal[c], :]
        corr_ref = pearson_corr_refv1(traces, labels, models_used)
        for i, (cv, cvr) in enumerate(zip(corr[:, c, :], corr_ref)):
            assert np.allclose(
                cv, cvr
            ), "[{}-{}-seed:{}-ns:{}-nc:{}-n:{}-nv:{}]\ncorr\n{}\nref\n{}".format(
                intermediate, c, seed, ns, nc, n, nv, cv, cvr
            )


def xor_pintern(nc):
    perm_internal = [[0 for _ in range(nc)] for _ in range(nc)]
    for i in range(nc):
        for j in range(nc):
            perm_internal[i][j] = j ^ i
    return perm_internal


def modadd_pintern(nc):
    perm_internal = [[0 for _ in range(nc)] for _ in range(nc)]
    for i in range(nc):
        for j in range(nc):
            perm_internal[i][j] = (j + i) % nc
    return perm_internal


def test_cpa_full():
    # CPA configuration
    cfgs = [(Cpa.Xor, xor_pintern), (Cpa.Add, modadd_pintern)]
    # Parameter space to test
    # (seed, ns, nc, n, nv)
    cases = [
        (0, 1, 2, 10, 1),
        (0, 1000, 2, 10, 1),
        (0, 1, 256, 1000, 1),
        (0, 1000, 256, 1000, 1),
        (0, 1, 2, 10, 2),
        (0, 1000, 256, 1000, 5),
    ]

    for cpa_intern, fn_pintern in cfgs:
        for seed, ns, nc, n, nv in cases:
            pintern = fn_pintern(nc)
            cpa_inner_intermediate(seed, ns, nc, n, nv, pintern, cpa_intern)
