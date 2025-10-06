import pytest
from scalib.postprocessing import mrs_gerber_lemma
import numpy as np


def test_mgl():

    err = 10**-15
    x = np.random.uniform(low=0, high=1)

    # Test with a single share
    assert abs(mrs_gerber_lemma(x, 2**8, base=2) - x) <= err

    # Test when a share leak more than a bit and another one less than a bit
    assert abs(mrs_gerber_lemma([x, 1.1], 2**8, base=2) - x) <= err

    # Test when two shares leaks between 1 and 2 bits
    assert (
        abs(
            mrs_gerber_lemma([1 + x, 1.1], 2**8, base=2)
            - (1 + mrs_gerber_lemma([x, 0.1], 2**8, base=2))
        )
        <= err
    )

    # Test when a single share is bellow noise amplification
    assert abs(mrs_gerber_lemma(x, 2**8 - 1, base=2) - x) <= err

    # Test when one share is bellow noise amplification threshold
    assert abs(mrs_gerber_lemma([x, 15], 2**8 - 1, base=2) - x) <= err

    # Test when no share is bellow noise amplification threshold
    assert abs(mrs_gerber_lemma([12, 15], 2**8 - 1, base=2) - 12) <= err
