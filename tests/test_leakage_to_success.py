import pytest
import numpy as np

from scalib.postprocessing import (
    success_rate,
    guessing_entropy,
    log_guessing_entropy,
    median,
)


def test_succss_rate():

    rng = np.random.default_rng(seed=42)

    key_size = 128

    mi = rng.uniform(low=0, high=key_size, size=10**3)

    # Check bound for typical key lenght
    sr_32 = success_rate(mi, key_size, enumeration_effort=32)
    assert (-key_size <= sr_32).all()
    assert (sr_32 <= 0).all()

    # Check that sr increase with enumeration effort
    sr_64 = success_rate(mi, key_size, enumeration_effort=64)
    assert (sr_64 >= sr_32).all()

    # Check wether it works for large keys
    key_size = 1024
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    sr = success_rate(mi, key_size, enumeration_effort=32)
    assert (-key_size <= sr).all()
    assert (sr <= 0).all()


def test_guessing_entropy():

    rng = np.random.default_rng(seed=42)

    # Check bound for typical key length
    key_size = 128
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    ge = guessing_entropy(mi, key_size)
    assert (ge >= 0).all()
    assert (ge <= key_size).all()

    # Check wether it works for large keys
    key_size = 1024
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    ge = guessing_entropy(mi, key_size)
    assert (ge >= 0).all()
    assert (ge <= key_size).all()


def test_log_guessing_entropy():

    rng = np.random.default_rng(seed=42)

    # Check wether it works for typical key lenght
    key_size = 128
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    lg = log_guessing_entropy(mi, key_size)
    assert (lg >= 0).all()
    assert (lg <= key_size).all()

    # Check wether it works for large keys
    key_size = 1024
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    lg = log_guessing_entropy(mi, key_size)
    assert (lg >= 0).all()
    assert (lg <= key_size).all()


def test_median():

    rng = np.random.default_rng(seed=42)

    # Check wether it works for typical key lenght
    key_size = 128
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    med = median(mi, key_size)
    assert (med >= 0).all()
    assert (med <= key_size).all()

    # Check wether it works for large keys
    key_size = 1024
    mi = rng.uniform(low=0, high=key_size, size=10**3)
    med = median(mi, key_size)
    assert (med >= 0).all()
    assert (med <= key_size).all()
