"""
Misc. internal SCALib utils.
"""

import contextlib
import signal
import threading

import numpy as np
import numpy.typing as npt


@contextlib.contextmanager
def interruptible():
    """Replace current SIGINT handler with OS-default one.

    This allows long-running non-python code (or multi-threaded one) to be
    interrupted. This results in unclean python shutdown but it is better than
    requiring to kill the process.

    This is only feasable on the main thread. In other threads, this function
    is a no-op.
    """
    if threading.current_thread() is threading.main_thread():
        restore_sig = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, restore_sig)
    else:
        yield


def assert_traces(l: npt.NDArray[np.int16], ns=None):
    if not isinstance(l, np.ndarray):
        raise ValueError(f"The traces is not a numpy array. {type(l)}")
    elif l.dtype != np.int16:
        raise ValueError(f"The traces array has dtype {l.dtype}, expected np.int16.")
    elif len(l.shape) != 2:
        raise ValueError(f"The traces array has {len(l.shape)} dimensions, expected 2.")
    elif ns is not None and l.shape[1] != ns:
        raise ValueError(
            f"Traces length {l.shape[1]} does not match previously-fitted traces ({ns})."
        )
    elif not l.flags.c_contiguous:
        raise ValueError(
            "Expected l to be a contiguous (C memory order) array. "
            "Use np.ascontiguous to change memory representation."
        )


def assert_classes(x: npt.NDArray[np.uint16], nv=None, multi=True):
    expected_dim = 2 if multi else 1
    if not isinstance(x, np.ndarray):
        raise ValueError("The classes is not a numpy array.")
    elif x.dtype != np.uint16:
        raise ValueError(f"The classes array has dtype {x.dtype}, expected np.uint16.")
    elif len(x.shape) != expected_dim:
        raise ValueError(
            f"The classes array has {len(x.shape)} dimensions, expected {expected_dim}."
        )
    elif multi and nv is not None and x.shape[1] != nv:
        raise ValueError(
            f"Number of variables {x.shape[1]} does not match  previously-fitted classes ({nv})."
        )
