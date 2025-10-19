import pytest
from scalib.postprocessing import mgl
import numpy as np


def test_mgl():

    rng = np.random.default_rng(seed=42)

    err = 10**-15
    x = rng.uniform(low=0, high=1)

    # Test the extreme case with group order of 2
    # Test with a single share
    assert abs(mgl(x, 2, base=2) - x) <= err

    # Verify that a value error is raised here
    with pytest.raises(ValueError):
        mgl([x, 1.1], 2, base=2)

    with pytest.raises(ValueError):
        mgl([1 + x, 1.1], 2, base=2)

    # Test with a single share
    assert abs(mgl(x, 2**8, base=2) - x) <= err

    # Test when a share leak more than a bit and another one less than a bit
    assert abs(mgl([x, 1.1], 2**8, base=2) - x) <= err

    # Test when two shares leaks between 1 and 2 bits
    assert (
        abs(mgl([1 + x, 1.1], 2**8, base=2) - (1 + mgl([x, 0.1], 2**8, base=2))) <= err
    )

    # Test when a single share is bellow noise amplification
    assert abs(mgl(x, 2**8 - 1, base=2) - x) <= err

    # Test when one share is bellow noise amplification threshold
    assert abs(mgl([x, 5], 2**8 - 1, base=2) - x) <= err

    # Test when no share is bellow noise amplification threshold
    assert abs(mgl([4, 5], 2**8 - 1, base=2) - 4) <= err

    # Test larger group order e.g. 3329 which the filed size in Kyber
    q = 3329

    # Test when a single share is bellow noise amplification
    assert abs(mgl(x, q, base=2) - x) <= err

    # Test when one share is bellow noise amplification threshold
    assert abs(mgl([x, 6], q, base=2) - x) <= err

    # Test when no share is bellow noise amplification threshold
    assert abs(mgl([5, 6], q, base=2) - 5) <= err
