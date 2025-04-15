import hashlib
import inspect
import numpy as np


def get_rng(**args):
    """
    Hash caller name (i.e. test name) to get the rng seed.
    args are also hashed in the seed.
    """
    # Use a deterministic hash (no need for cryptographic robustness, but
    # python's hash() is randomly salted).
    seed = hashlib.sha256((repr(args) + inspect.stack()[1][3]).encode()).digest()
    return np.random.Generator(np.random.PCG64(list(seed)))
