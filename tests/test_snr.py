import pytest
from scalib.metrics import SNR
import numpy as np


def test_snr():
    """
    Test the SNR.
    """
    nc = 16
    nv = 4
    ns = 10
    n = 1000
    batch = 100
    x = np.random.randint(0, 256, (n, ns), dtype=np.int16) - 128
    y = np.random.randint(0, nc, (n, nv), dtype=np.uint16)

    # compute SNR with SCALib
    snr = SNR(np=nv, nc=nc, ns=ns)
    for i in range(0, n, batch):
        snr.fit_u(x[i : i + batch, :], y[i : i + batch, :])
    snr_val = snr.get_snr()

    # compte SNR for each method
    means = np.zeros((nc, ns))
    vars = np.zeros((nc, ns))
    for v in range(nv):
        for c in range(nc):
            means[c] = np.mean(x[np.where(y.transpose()[v] == c)[0], :], axis=0)
            vars[c] = np.var(x[np.where(y.transpose()[v] == c)[0], :], axis=0)
        snr_ref = np.var(means, axis=0) / np.mean(vars, axis=0)
        assert np.allclose(snr_ref, snr_val[v])
