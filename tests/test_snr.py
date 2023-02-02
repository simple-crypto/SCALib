import pytest
from scalib import ScalibError
from scalib.metrics import SNR
import numpy as np


def gen_snr_data(nv, nc, ns, n):
    ii16 = np.iinfo(np.int16)
    x = np.random.randint(ii16.min, ii16.max, (n, ns), dtype=np.int16)
    x[0, 0] = ii16.min
    if n > 1:
        x[1, 0] = ii16.max
    y = np.random.randint(0, nc, (n, nv), dtype=np.uint16)
    return x, y


def snr_pooled(x, y, nv, nc, ns):
    snr_ref = np.zeros((nv, ns), dtype=np.float64)
    g_mean = np.mean(x, axis=0)
    for v in range(nv):
        sig_acc = np.zeros((ns,))
        noise_acc = np.zeros((ns,))
        for c in range(nc):
            samples = x[np.where(y.transpose()[v] == c)[0], :]
            if samples.shape[0] != 0:
                mean = np.mean(samples, axis=0)
                sig_acc += (mean - g_mean) ** 2 * samples.shape[0]
                noise_acc += ((samples - mean[np.newaxis, :]) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_ref[v, :] = (sig_acc / x.shape[0]) / (noise_acc / x.shape[0])
    return snr_ref


def snr_pooled2(x, y, nv, nc, ns, use_64bit):
    """Should be almost exact implementation of rust version, including rounding
    errors."""

    if use_64bit:
        stype = np.int64
    else:
        stype = np.int32

    sig = np.zeros((nv, ns), dtype=np.float64)
    no = np.zeros((nv, ns), dtype=np.float64)
    g_sum = np.sum(x.astype(np.int64), axis=0).astype(object)
    sum_sq = np.sum(x.astype(np.int64) ** 2, axis=0).astype(object)

    n = x.shape[0]
    for v in range(nv):
        sum_sq_cls = np.zeros((ns,)).astype(object)
        for c in range(nc):
            samples = x[np.where(y.transpose()[v] == c)[0], :]
            if samples.shape[0] != 0:
                s = np.sum(samples.astype(stype), axis=0).astype(object)
                sum_sq_cls += (s**2).astype(object) * n // samples.shape[0]
        sig[v, :] = (sum_sq_cls - g_sum**2).astype(np.float64)
        assert n == x.shape[0]
        no[v, :] = (x.shape[0] * sum_sq - sum_sq_cls).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_ref = sig / no
    return snr_ref


def snr_run(nc, nv, ns, n, use_64bit, batch=100):
    """
    Test the SNR.
    """
    print(f"SNR run {nc=} {nv=} {ns=} {n=} {use_64bit=} {batch=}")
    x, y = gen_snr_data(nv, nc, ns, n)

    # compute SNR with SCALib
    snr = SNR(np=nv, nc=nc, ns=ns, use_64bit=use_64bit)

    n_samples = np.array([[(y[:, v] == c).sum() for c in range(nc)] for v in range(nv)])

    if (n_samples >= (2**16)).any() and not use_64bit:
        with pytest.raises(Exception):
            for i in range(0, n, batch):
                snr.fit_u(x[i : i + batch, :], y[i : i + batch, :])
            snr_val = snr.get_snr()
        return

    for i in range(0, n, batch):
        snr.fit_u(x[i : i + batch, :], y[i : i + batch, :])
    snr_val = snr.get_snr()

    snr_ref = snr_pooled2(x, y, nv, nc, ns, use_64bit)
    assert np.allclose(snr_ref, snr_val, equal_nan=True)
    snr_ref = snr_pooled(x, y, nv, nc, ns)
    if not (n_samples == 0).any():
        assert np.allclose(snr_ref, snr_val, equal_nan=True, rtol=1 / n, atol=0.1 / n)


def test_snr():
    test_cases = [
        {"nc": 2, "nv": 1, "ns": 1, "n": 1},
        {"nc": 2, "nv": 1, "ns": 1, "n": 4},
        {"nc": 2, "nv": 1, "ns": 10, "n": 4},
        {"nc": 2, "nv": 1, "ns": 1, "n": 10},
        {"nc": 2, "nv": 1, "ns": 1, "n": 1000},
        {"nc": 16, "nv": 5, "ns": 10, "n": 1000},
        {"nc": 16, "nv": 32, "ns": 10, "n": 10},
        {"nc": 16, "nv": 153, "ns": 10, "n": 10},
        {"nc": 1024, "nv": 3, "ns": 10, "n": 10},
        {"nc": 2, "nv": 3, "ns": 10**4, "n": 10},
        {"nc": 2, "nv": 1, "ns": 1, "n": 1 << 17},
    ]
    for test_case in test_cases:
        for use_64bit in [False, True]:
            snr_run(use_64bit=use_64bit, **test_case)


def test_large_snr():
    # Was deadlocking on error (only when displaying progress bar, i.e., for
    # large cases)
    ntraces = 1000
    ns = 10000
    nv = 1000
    traces = np.random.randint(0, 256, (ntraces, ns), dtype=np.int16)
    nb = 16
    X = np.random.randint(0, nb, (ntraces, nv), dtype=np.uint16)
    snr = SNR(nb, ns, nv)
    snr.fit_u(traces, X)
    snr.get_snr()

    snr = SNR(nb - 1, ns, nv)
    with pytest.raises(ScalibError):
        snr.fit_u(traces, X)


def test_extreme_nc():
    # Issue #54
    for nc in (1, 2**16 + 1):
        with pytest.raises(Exception):
            SNR(nc, 10, 1)
    for nc in (2, 2**16):
        SNR(nc, 10, 1)
