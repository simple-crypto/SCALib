"""
    This script feeds MTtest object and compare the outcome with the reference
    implementation. 
    
    The tests covers:
        - various d
        - Usage of incremental API. 
        - Number of pois
        - Number of traces
"""
import pytest
from scalib.metrics import MTtest
import numpy as np


def reference(traces, x, d, pois):
    I0 = np.where(x == 0)[0]
    I1 = np.where(x == 1)[0]

    n0 = len(I0)
    n1 = len(I1)

    t0 = traces[I0]
    t1 = traces[I1]

    t0 = t0 - np.mean(t0, axis=0)
    t1 = t1 - np.mean(t1, axis=0)
    if d > 2:
        t0 = t0 / np.std(t0, axis=0)
        t1 = t1 / np.std(t1, axis=0)

    t0_mult = 1
    t1_mult = 1
    for d in range(d):
        t0_mult *= t0[:, pois[d, :]]
        t1_mult *= t1[:, pois[d, :]]

    print(t0_mult.shape)
    u0 = np.mean(t0_mult, axis=0)
    u1 = np.mean(t1_mult, axis=0)

    v0 = np.var(t0_mult, axis=0)
    v1 = np.var(t1_mult, axis=0)

    return (u0 - u1) / np.sqrt((v0 / n0) + (v1 / n1))


def test_ttest_d2_many_3():
    ns = 10237
    npois = 12342
    d = 2
    nc = 2
    n = 1 << 8

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)
    traces += m[labels]

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d2_many_2():
    ns = 99
    npois = 12329
    d = 2
    nc = 2
    n = 1 << 8

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)
    traces += m[labels]

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d2_many():
    ns = 10
    npois = 6
    d = 2
    nc = 2
    n = 1 << 17

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)
    traces += m[labels]

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d2():
    ns = 100
    npois = 6302
    d = 2
    nc = 2
    n = 200

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)
    traces += m[labels]

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d3():
    ns = 100
    npois = 10
    d = 3
    nc = 2
    n = 200

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)
    traces += m[labels]

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    ttest.fit_u(traces, labels)
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d3_multiple_fit():
    ns = 1043
    d = 3
    nc = 2
    npois = 10021
    n = 40

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    for i in range(0, n, 10):
        ttest.fit_u(traces[i : i + 10, :], labels[i : i + 10])
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d3_multiple_fit():
    ns = 1043
    d = 3
    nc = 2
    npois = 10021
    n = 40

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(0, 10, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    for i in range(0, n, 10):
        ttest.fit_u(traces[i : i + 10, :], labels[i : i + 10])
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)


def test_ttest_d2_multiple_fit():
    ns = 1043
    d = 2
    nc = 2
    npois = 10021
    n = 40

    m = np.random.randint(0, 2, (nc, ns))
    traces = np.random.randint(-4231, 4214, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    pois = np.random.randint(0, ns, (d, npois), dtype=np.uint32)

    t_ref = reference(traces, labels, d, pois)
    ttest = MTtest(d, pois)
    for i in range(0, n, 10):
        ttest.fit_u(traces[i : i + 10, :], labels[i : i + 10])
    t = ttest.get_ttest()
    assert np.allclose(t_ref, t, rtol=1e-3)
