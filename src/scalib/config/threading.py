"""Configuration of SCALib's threadpool."""

import os

from scalib import _scalib_ext


class ThreadPool:
    """SCALib threadpool.

    All computationally-heavy operations of SCALib are run on a thread pool.
    To use the configured `ThreadPool`, use `thread_context` or
    `default_threadpool`.
    """

    def __init__(self, n_threads):
        self.n_threads = n_threads
        self._pool = None

    @property
    def pool(self):
        # We initialize the true threadpool lazily for 2 reasons:
        # - It avoids creating threads when importing SCALib, which reduces
        # import time and, more importantly, prevents bugs with the usage of
        # subprocesses (such as with concurrent.futures.ProcessPoolExecutor),
        # since the POSIX API does not allow fork'ing after threadds have been
        # created.
        # - It generally improves performance when the threadpool is not actually used.
        if self._pool is None:
            self._pool = _scalib_ext.ThreadPool(self.n_threads)
        return self._pool


def _default_num_threads():
    num_threads = os.environ.get("SCALIB_NUM_THREADS")
    if num_threads is None:
        num_threads = _scalib_ext.usable_parallelism()
    else:
        try:
            num_threads = int(num_threads)
        except ValueError:
            raise ValueError(
                "Environment variable SCALIB_NUM_THREADS must be an integer."
            )
    return num_threads
