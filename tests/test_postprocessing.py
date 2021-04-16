import pytest
import numpy as np
from scalib.postprocessing import rank_accuracy


def test_rank_accuracy():
    nc = 256
    nsubkeys = 4
    acc = 0.2

    costs = np.zeros((nsubkeys, nc)) + 0.1
    secret_key = np.random.randint(0, nc, nsubkeys)
    costs[np.arange(nsubkeys), secret_key] = 1.0

    rmin, r, rmax = rank_accuracy(-np.log10(costs), secret_key, acc_bit=acc)

    assert r == 1.0
    assert np.log2(rmax) - np.log2(rmin) <= acc
