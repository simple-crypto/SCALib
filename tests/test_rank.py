import pytest
import numpy as np

from scalib.postprocessing import rank_accuracy


def test_many_subkeys():
    KEY_SIZE = 512
    COEFF_SIZE = 5

    key = np.random.randint(0, COEFF_SIZE, KEY_SIZE)

    scores = np.ones((KEY_SIZE, COEFF_SIZE)) * 1e-5
    scores[np.arange(KEY_SIZE), key] = 1

    (rmin, r, rmax) = rank_accuracy(-np.log(scores), key)
    assert rmin == 1
    assert rmax == 1
    assert r == 1
