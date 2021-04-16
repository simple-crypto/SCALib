import pytest
from scalib.metrics import Ttest
import numpy as np
import scipy.stats


def reference(traces, x, D):

    CM0 = np.zeros((D * 2, traces.shape[1]))
    I0 = np.where(x == 0)[0]
    u0 = np.mean(traces[I0, :], axis=0)
    for d in range(D * 2):
        CM0[d, :] = np.mean((traces[I0, :] - u0) ** (d + 1), axis=0)

    CM1 = np.zeros((D * 2, traces.shape[1]))
    I1 = np.where(x == 1)[0]
    u1 = np.mean(traces[I1, :], axis=0)
    for d in range(D * 2):
        CM1[d, :] = np.mean((traces[I1, :] - u1) ** (d + 1), axis=0)
    u1_ref = u1
    u0_ref = u0
    t = np.zeros((D, len(traces[0, :])))
    n = [len(I0), len(I1)]
    for d in range(1, D + 1):
        if d == 1:
            u0 = u0_ref
            u1 = u1_ref
            v0 = CM0[1, :]
            v1 = CM1[1, :]
        elif d == 2:
            u0 = CM0[1, :]
            u1 = CM1[1, :]
            v0 = CM0[3, :] - CM0[1, :] ** 2
            v1 = CM1[3, :] - CM1[1, :] ** 2
        else:
            u0 = CM0[d - 1, :] / np.power(CM0[1, :], d / 2)
            u1 = CM1[d - 1, :] / np.power(CM1[1, :], d / 2)
            v0 = (CM0[(d * 2) - 1, :] - CM0[(d) - 1, :] ** 2) / (CM0[1, :] ** d)
            v1 = (CM1[(d * 2) - 1, :] - CM1[(d) - 1, :] ** 2) / (CM1[1, :] ** d)

        t[d - 1, :] = (u0 - u1) / (np.sqrt((v0 / n[0]) + (v1 / n[1])))
    return t


def test_ttest_d1():
    ns = 100
    d = 1
    nc = 2
    n = 200

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    t_ref = reference(traces, labels, d)
    ttest = Ttest(ns, 1)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d2():
    ns = 100
    d = 2
    nc = 2
    n = 200

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    t_ref = reference(traces, labels, d)
    ttest = Ttest(ns, d)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d6_multiple_fit():
    ns = 100
    d = 6
    nc = 2
    n = 200

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    t_ref = reference(traces, labels, d)
    ttest = Ttest(ns, d)
    for i in range(0, n, 10):
        ttest.fit_u(traces[i : i + 10, :], labels[i : i + 10])
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d6():
    ns = 100
    d = 6
    nc = 2
    n = 200

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    t_ref = reference(traces, labels, d)
    ttest = Ttest(ns, d)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)
