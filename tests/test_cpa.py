import pytest
import numpy as np
import hashlib
import inspect

from scalib.attacks import CPA


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


def numpy_corr(x, y):
    corrmat = np.corrcoef(x, y)
    ns = x.shape[0]
    extract = [corrmat[i, j] for i, j in zip(range(ns), range(ns, ns + ns))]
    return np.array(extract)


def hw(v, nbits):
    return sum([(v >> i) & 0b1 for i in range(nbits)])


HW = np.array([hw(e, 8) for e in range(256)], dtype=np.int16)


def test_cpa_univariate_correlation():
    ns = 5
    nbits = 1
    nc = 2**nbits
    n = 10

    rng = get_rng()
    traces = rng.integers(0, 10, (n, ns), dtype=np.int16)
    labels = rng.integers(0, nc, (n, 1), dtype=np.uint16)

    print("labels")
    print(labels)
    print()

    ### Create the CPA and fit all with scalib
    cpa = CPA(nc)
    cpa.fit_u(traces, labels)

    models = np.hstack([HW[np.arange(nc)][:, np.newaxis] for _ in range(ns)]).astype(
        np.float64
    )[np.newaxis, :, :]
    print("SCAlib models")
    print(models)

    corr_scalib = cpa.get_correlation(models, CPA.Intermediate.XOR)
    corr = corr_scalib[0, 0, :]

    ### CPA ref
    models_ref = np.vstack([np.array(ns * [HW[e]]).reshape([1, ns]) for e in labels])
    corr_ref = pearson_corr(traces, models_ref)
    print("Ref models")
    print(models_ref)
    print()

    print(corr.shape)
    print(corr_ref.shape)

    print(corr)
    print(corr_ref)

    npcorr = numpy_corr(traces.T, models_ref.T)
    print(np.allclose(npcorr, corr_ref))

    assert np.allclose(corr, corr_ref)
