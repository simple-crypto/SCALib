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


def clean_traces(traces: npt.NDArray[np.int16], ns=None):
    if not isinstance(traces, np.ndarray):
        raise ValueError(f"The traces is not a numpy array. {type(traces)}")
    elif traces.dtype != np.int16:
        raise ValueError(
            f"The traces array has dtype {traces.dtype}, expected np.int16."
        )
    elif len(traces.shape) != 2:
        raise ValueError(
            f"The traces array has {len(traces.shape)} dimensions, expected 2."
        )
    elif ns is not None and traces.shape[1] != ns:
        raise ValueError(
            f"Traces length {traces.shape[1]} does not match previously-fitted traces ({ns})."
        )
    return np.ascontiguousarray(traces)


def clean_labels(x: npt.NDArray, nv=None, multi=True, exp_type=np.uint16):
    expected_dim = 2 if multi else 1
    if not isinstance(x, np.ndarray):
        raise ValueError("The classes is not a numpy array.")
    elif x.dtype != exp_type:
        raise ValueError(f"The classes array has dtype {x.dtype}, expected {exp_type}.")
    elif len(x.shape) != expected_dim:
        raise ValueError(
            f"The classes array has {len(x.shape)} dimensions, expected {expected_dim}."
        )
    elif multi and nv is not None and x.shape[1] != nv:
        raise ValueError(
            f"Number of variables {x.shape[1]} does not match  previously-fitted classes ({nv})."
        )
    return x
