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


def test_run_hwlda():
    nb = 4
    nv = 1
    n = 100

    bits_data = generate_random_data_bits(n, nv, nb)
    hw_nfree = generate_noiseless_HW_traces_mvars(bits_data)
    data_u64 = u64_data_from_bits(bits_data)

    hwldaacc = HwLdaAcc(nb)
    hwldaacc.fit_u(hw_nfree, data_u64)

    hwlda = HwLda(hwldaacc)
    proba = hwlda.predict_proba(hw_nfree, 0)

    for i in range(n):
        print(f'{i} ; {hw_nfree[i,0]} -  {proba[i, hw_nfree[i,0]]}')

    print(proba.shape)

