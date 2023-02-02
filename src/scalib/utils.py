"""
Misc. internal SCALib utils.
"""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import contextvars
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


class ContextExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with automatic handling of contextvars."""

    def __init__(self, *args, **kwargs):
        self.context = contextvars.copy_context()
        super().__init__(*args, **kwargs, initializer=self._set_child_context)

    def _set_child_context(self):
        for var, value in self.context.items():
            var.set(value)
