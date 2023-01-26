"""
Misc. internal SCALib utils.
"""

import contextlib
import signal
import threading


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
