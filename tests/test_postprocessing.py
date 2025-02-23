import pytest
import numpy as np
import random
from scalib.postprocessing import rank_accuracy

static_probs = [
    [0.5, 0.25, 0.25, 0.125],
    [0.5, 0.25, 0.125, 0.25],
    [0.5, 0.5, 0.125, 0.33333333],
    [0.33333333, 0.25, 0.5, 0.33333333],
    [0.33333333, 0.25, 0.33333333, 0.25],
    [0.33333333, 0.25, 0.5, 0.125],
    [0.33333333, 0.125, 0.25, 0.5],
    [0.125, 0.33333333, 0.33333333, 0.25],
    [0.25, 0.5, 0.125, 0.25],
    [0.125, 0.125, 0.5, 0.25],
    [0.125, 0.5, 0.25, 0.125],
    [0.33333333, 0.25, 0.25, 0.25],
    [0.125, 0.25, 0.25, 0.125],
    [0.25, 0.33333333, 0.25, 0.33333333],
    [0.25, 0.125, 0.5, 0.33333333],
    [0.125, 0.25, 0.125, 0.25],
]


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



#NTL would be needed for that
"""
# Compare the ntl and scaled histogram implementation with a normal probability distribution
def test_rank_accuracy_scaled_vs_ntl():
    nc = 256
    nsubkeys = 16
    max_error = 0.5
    k_probs = np.zeros((nsubkeys, nc))

    secret_key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for j in range(nsubkeys):
        for i in range(nc):
            k_probs[j][i] = 1 / random.randint(2, 20)

    rmin, r, rmax = rank_accuracy(-np.log10(k_probs), secret_key, method="histbignum")
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    rmin, r, rmax = rank_accuracy(-np.log10(k_probs), secret_key, method="scaledhist")
    lrmin_scaled, lr_scaled, lrmax_scaled = (np.log2(rmin), np.log2(r), np.log2(rmax))

    assert np.abs(lrmin - lrmin_scaled) <= max_error
    assert np.abs(lr - lr_scaled) <= max_error
    assert np.abs(lrmax - lrmax_scaled) <= max_error


# Compare the ntl and scaled histogram implementation in rand edge cases
# The normal histogram implementation would most likely return negative ranks
def test_rank_accuracy_scaled_edge_cases():
    nc = 256
    nsubkeys = 16
    max_error = 3.0
    k_probs = np.zeros((nsubkeys, nc))
    secret_key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for j in range(nsubkeys):
        for i in range(nc):
            if i < 6:
                k_probs[j][i] = 1 / random.randint(2, 5)
            else:
                k_probs[j][i] = 1 / 16

    rmin, r, rmax = rank_accuracy(-np.log10(k_probs), secret_key, method="histbignum")
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    rmin, r, rmax = rank_accuracy(
        -np.log10(k_probs), secret_key, method="scaledhist", acc_bit=7.0
    )
    lrmin_scaled, lr_scaled, lrmax_scaled = (np.log2(rmin), np.log2(r), np.log2(rmax))
    rmin, r, rmax = rank_accuracy(-np.log10(k_probs), secret_key)
    assert np.abs(lrmin - lrmin_scaled) <= max_error
"""

# Compare the ntl and scaled histogram implementation in a known edge case
# The normal histogram implementation would return negative ranks
def test_rank_accuracy_scaled_edge_case():
    max_error = 3.0  # allowed rounding error
    # reference value from the ntl execution
    ntl_low = 52.40236985817056
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

    # Using lower and upper histograms may impact accuracy (space between min and max rank) in some cases
    # a high acc_bit helps to prevent infinite loops and show the actual benefit of this method
    rmin, r, rmax = rank_accuracy(
        -np.log10(k_probs), secret_key, method="scaledhist", acc_bit=10.0
    )
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    assert np.abs(lrmin - ntl_low) <= max_error
