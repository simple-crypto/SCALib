import pytest
from scalib.metrics import SNR
import numpy as np

def gen_snr_data(nv, nc, ns, n):
    x = np.random.randint(0, 256, (n, ns), dtype=np.int16) - 128
    y = np.random.randint(0, nc, (n, nv), dtype=np.uint16)
    return x, y

def snr_naive(x, y, nv, nc, ns):
    means = np.zeros((nc, ns))
    vars = np.zeros((nc, ns))
    snr_ref = np.zeros((nv, ns), dtype=np.float64)
    for v in range(nv):
        for c in range(nc):
            means[c] = np.mean(x[np.where(y.transpose()[v] == c)[0], :], axis=0)
            vars[c] = np.var(x[np.where(y.transpose()[v] == c)[0], :], axis=0)
        snr_ref[v,:] = np.var(means, axis=0) / np.mean(vars, axis=0)
    return snr_ref

def snr_pooled(x, y, nv, nc, ns):
    snr_ref = np.zeros((nv, ns), dtype=np.float64)
    g_mean = np.mean(x, axis=0)
    for v in range(nv):
        sig_acc = np.zeros((ns,))
        noise_acc = np.zeros((ns,))
        for c in range(nc):
            samples = x[np.where(y.transpose()[v] == c)[0], :]
            mean = np.mean(samples, axis=0)
            sig_acc += (mean-g_mean)**2*samples.shape[0]
            noise_acc += ((samples-mean[np.newaxis,:])**2).sum(axis=0)
        snr_ref[v,:] = (sig_acc / x.shape[0]) / (noise_acc / x.shape[0])
    return snr_ref

def test_snr():
    """
    Test the SNR.
    """
    nc = 16
    nv = 4
    ns = 10
    n = 1000
    batch = 100
    x, y = gen_snr_data(nv, nc, ns, n)

    # compute SNR with SCALib
    snr = SNR(np=nv, nc=nc, ns=ns)
    for i in range(0, n, batch):
        snr.fit_u(x[i : i + batch, :], y[i : i + batch, :])
    snr_val = snr.get_snr()

    snr_ref = snr_naive(x, y, nv, nc, ns)
    assert np.allclose(snr_ref, snr_val)

def test_snr2():
    """
    Test the SNR.
    """
    nc = 16
    nv = 4
    ns = 10
    n = 1000
    batch = 100
    x, y = gen_snr_data(nv, nc, ns, n)

    # compute SNR with SCALib
    snr = SNR(np=nv, nc=nc, ns=ns, use_newer=True)
    for i in range(0, n, batch):
        snr.fit_u(x[i : i + batch, :], y[i : i + batch, :])
    snr_val = snr.get_snr()

    snr_ref = snr_pooled(x, y, nv, nc, ns)
    assert np.allclose(snr_ref, snr_val)
