import pytest
import numpy as np
from scalib.postprocessing import rank_accuracy

static_probs = ([[0.5,    0.25,   0.25,   0.125],
                 [0.5,    0.25,   0.125,  0.25],
                 [0.5,    0.5,    0.125,  0.33333333],
                 [0.33333333, 0.25,   0.5,    0.33333333],
                 [0.33333333, 0.25,   0.33333333, 0.25],
                 [0.33333333, 0.25,   0.5,    0.125],
                 [0.33333333, 0.125,  0.25,   0.5],
                 [0.125,  0.33333333, 0.33333333, 0.25],
                 [0.25,   0.5,    0.125,  0.25],
                 [0.125,  0.125,  0.5,    0.25],
                 [0.125,  0.5,    0.25,   0.125],
                 [0.33333333, 0.25,   0.25,   0.25],
                 [0.125,  0.25,   0.25,   0.125],
                 [0.25,   0.33333333, 0.25,   0.33333333],
                 [0.25,   0.125,  0.5,    0.33333333],
                 [0.125,  0.25,   0.125,  0.25]])

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


def test_rank_accuracy2():
    max_error = 0.2 #allowed rounding error
    #reference values from the ntl execution
    ntl_low = 52.40236985817056
    ntl_mid = 52.45835852381074
    ntl_high = 52.70108474865651

    nc = 256
    nsubkeys = 16

    k_probs = np.zeros((nsubkeys, nc))
    secret_key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for j in range(nsubkeys):
        for i in range(nc):
            if i < 4:
                k_probs[j][i] = static_probs[j][i]
            else:
                k_probs[j][i] = 1 / 16

    
    rmin, r, rmax = rank_accuracy(-np.log10(k_probs), secret_key, method="hist")

    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))

    assert np.abs(lrmin - ntl_low) <= max_error
    assert np.abs(lr - ntl_mid) <= max_error
    assert np.abs(lrmax - ntl_high) <= max_error