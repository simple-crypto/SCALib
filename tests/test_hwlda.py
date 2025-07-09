import pickle

import pytest
import numpy as np
import scipy.linalg
import scipy.special

from scalib import ScalibError
from scalib.modeling import HwLda, HwLdaAcc

from utils_test import get_rng

import copy

# Table for simple HW
HW = np.sum(
    np.unpackbits(np.arange(256, dtype=np.uint8), bitorder="little").reshape((256, 8)),
    axis=1,
)


def generate_random_data_bits(n, nv, nb):
    """
    Return array of shape (n, nv, nb)
    """
    return np.random.randint(0, 2, (n, nv, nb), dtype=np.uint8)


def u64_data_from_bits(data_bits):
    (n, nv, nb) = data_bits.shape
    u64d = np.zeros((n, nv), dtype=np.uint64)
    for nv in range(nv):
        for bi in range(nb):
            u64d[:, nv] |= data_bits[:, nv, bi] << bi
    return u64d


def generate_noiseless_HW_traces_mvars(data_bits):
    return np.sum(data_bits, axis=2).astype(np.int16)


def generate_noisy_scaled_hw_mv(dbits, nstd, scale=1 << 10):
    nv = dbits.shape[1]
    noiseless_hw = generate_noiseless_HW_traces_mvars(dbits)
    noise = np.random.normal(0, nstd, size=noiseless_hw.shape)
    traces = np.round(((noiseless_hw + noise) * scale)).astype(np.int16)
    return traces, noiseless_hw


def test_run_hwlda_mvars_lownoise():
    nb = 8
    nv = 3
    n = 10000
    nstd = 0.1

    SCALE = 1 << 10

    # data generation for training
    dbits = generate_random_data_bits(n, nv, nb)
    traces, hw = generate_noisy_scaled_hw_mv(dbits, nstd, scale=SCALE)
    du64 = u64_data_from_bits(dbits)

    # Accumulator
    hwldaacc = HwLdaAcc(nb)
    hwldaacc.fit_u(traces, du64)

    # Solve
    hwlda = HwLda(hwldaacc)

    # Data generation for prediction
    nval = 20
    pdbits = generate_random_data_bits(nval, nv, nb)
    ptraces, phw = generate_noisy_scaled_hw_mv(pdbits, nstd)
    pdu64 = u64_data_from_bits(pdbits)

    # HW probas
    hwprobaas = hwlda.predict_hw_probas(ptraces)
    lprobas = hwlda.predict_log2p1(ptraces,pdu64)

    for vi in range(nv):
        # Prediction classes
        proba = hwlda.predict_proba(ptraces, vi)
        mproba = np.max(proba, axis=1)
        for ni in range(nval):
            print(f'\n## n: {ni} ; vi: {vi}')
            print(f'L(x): {ptraces[ni,vi]} ; HW(x): {phw[ni,vi]} ; class: {pdu64[ni,vi]}')
            c = pdu64[ni,vi]
            prv = proba[ni,c]
            print(f'Pr class:"{c}": {prv} (max is {mproba[ni]})')
            assert np.allclose(prv, mproba[ni])
            lprob = np.log2(prv)
            print(f'lPr class:"{c}": {lprob} ; l2p1: {lprobas[ni, vi]}')
            assert np.allclose(lprobas[ni, vi], lprob), "Log2 failure"
            max_hwpr = np.argmax(hwprobaas[vi, ni])
            assert max_hwpr==phw[ni, vi] 

def test_run_hwlda_univariate_lownoise():
    nb = 4
    nv = 1
    n = 10000
    nstd = 0.1

    SCALE = 1 << 10
    assert nb <= 8
    hwmap = np.array([HW[e] for e in range(1 << nb)])

    # data generation for training
    dbits = generate_random_data_bits(n, nv, nb)
    traces, hw = generate_noisy_scaled_hw_mv(dbits, nstd, scale=SCALE)
    du64 = u64_data_from_bits(dbits)

    # Some stati
    thw = 2
    std_prat = np.std(traces[hw == thw])
    print(f"SCALED STD: {std_prat}")
    print(f"UNSCALED_STD:{std_prat/SCALE} (must be {nstd})")

    # Accumulator
    hwldaacc = HwLdaAcc(nb)
    hwldaacc.fit_u(traces, du64)

    # Solve
    hwlda = HwLda(hwldaacc)

    # Data generation for prediction
    nval = 20
    pdbits = generate_random_data_bits(nval, nv, nb)
    ptraces, phw = generate_noisy_scaled_hw_mv(pdbits, nstd)
    pdu64 = u64_data_from_bits(pdbits)

    # Prediction classes
    proba = hwlda.predict_proba(ptraces, 0)
    mproba = np.max(proba, axis=1)

    # Prediction HW
    hwprobaas = hwlda.predict_hw_probas(ptraces)[0]
    hw_mproba = np.max(hwprobaas, axis=1)

    # Log prob
    lprobas = hwlda.predict_log2p1(ptraces, pdu64)
    _lprobas = np.log2(proba)

    for i, (tr, hw, c, pr, mpr, hwpr, mhwpr, lprl, lpr) in enumerate(
        zip(ptraces, phw, pdu64, proba, mproba, hwprobaas, hw_mproba, lprobas, _lprobas)
    ):
        print(f"\n## CASE: {i} (scaling factor: {SCALE})")
        print(f"HW: {hw}, scaled noisy HW: {tr}, class:{c}")
        print(f"probas: {pr} (max proba {mpr})")
        print(f"proba of class {c}: {pr[c]}")
        # Validate the max probability
        assert np.allclose(pr[c], mpr), "Max probability not found."
        # Validate the HW
        print(f"HW probas: {hwpr} (max proba {mhwpr})")
        print(f"proba of HW {hw}: {hwpr[hw]}")
        assert np.allclose(hwpr[hw], mhwpr), "MAX HW not recovered"
        for hwt in range(nb + 1):
            pr_hwt = pr[hwmap == hwt]
            spr_hwt = np.sum(pr_hwt)
            assert np.allclose(
                spr_hwt, hwpr[hwt]
            ), "HW proba computation failure compared to the exhaustive classes"
        # Validate the log2proba
        print(f"logpr: {lpr}")
        print(f"logpr class {c}: {lpr[c]}")
        print(f"logpr: {lprl}")
        assert np.allclose(lpr[c], lprl), "Log proba failure"
