"""Configuration of SCALib's threadpool.

All computationally-heavy operations of SCALib are run on a thread pool.
The thread pools are  created with the `ThreadPool` class, and activated using
the `tread_context` context manager, or the `default_threadpool` function.

The initial number of threads in the default thread pool is determined by the
value of the environment variable `SCALIB_NUM_THREADS` if it is set such as:

.. code-block::

    SCALIB_NUM_THREADS=8 python3 XXX.py

If `SCALIB_NUM_THREADS` is not set, 
reasonable default for the typical use of SCALib is taken (it is currently the
one given 
`here <https://docs.rs/num_cpus/1.13.1/num_cpus/fn.get_physical.html>`_),
but might not be optimal for your workload.
For best performance, we recommend tuning this to your machine and workload,
where two good starting points are the number of logical or physical cores in
the system.

Examples
--------

The used `ThreadPool` can also be defined directly in the python scripts. 

>>> from scalib.config.threading import thread_context, ThreadPool
>>> # Example 1: Set a default ThreadPool to 10 threads.
>>> default_threadpool(10)
>>> # Example 2: All computation-intensive tasks of SCALib will be run on 5 threads.
>>> with thread_context(5):
...     pass
>>> # Example 3: One can also re-use a ThreadPool
>>> # If the two following statements are executed in parallel (on multiple
>>> # Python threads), both computations will share the same 5 threads.
>>> pool = ThreadPool(5)
>>> with thread_context(pool):
...     pass
>>> with thread_context(pool):
...     pass

"""

import contextvars
import contextlib
import os

from scalib import _scalib_ext


@contextlib.contextmanager
def thread_context(threads):
    """Locally activate a threadpool.

    Parameters
    ----------
    threads: ThreadPool or int
        Either `ThreadPool` to use, or use a fresh `ThreadPool` with `threads`
        threads.
    """
    pool = _threads_as_pool(threads)
    restore_token = _thread_pool.set(pool)
    yield pool
    _thread_pool.reset(restore_token)


def default_threadpool(threads):
    """Configure the default SCALib threadpool.

    This threadpool is used when no thread_context is active.

    Parameters
    ----------
    threads: ThreadPool or int
        Either `ThreadPool` to use, or use a fresh `ThreadPool` with `threads`
        threads.
    """
    pool = _threads_as_pool(threads)
    # Updating the _default_threadpool directly would not do anything.
    _default_threadpool.pool = pool.pool


class ThreadPool:
    """SCALib threadpool.

    All computationally-heavy operations of SCALib are run on a thread pool.
    To use the configured `ThreadPool`, use `thread_context` or
    `default_threadpool`.
    """

    def __init__(self, n_threads):
        self.n_threads = n_threads
        self.pool = _scalib_ext.ThreadPool(n_threads)


_num_threads = os.environ.get("SCALIB_NUM_THREADS")
if _num_threads is None:
    _num_threads = _scalib_ext.get_n_cpus_physical()
else:
    try:
        _num_threads = int(_num_threads)
    except ValueError:
        raise ValueError("Environment variable SCALIB_NUM_THREADS must be an integer.")
_default_threadpool = ThreadPool(_num_threads)
_thread_pool = contextvars.ContextVar("thread_pool", default=_default_threadpool)


def _threads_as_pool(threads):
    if isinstance(threads, int):
        pool = ThreadPool(threads)
    elif isinstance(threads, ThreadPool):
        pool = threads
    else:
        raise TypeError(f"threads is {type(threads)}, must be int or ThreadPool.")
    return pool


def _get_threadpool():
    """Get the current threadpool, to give to Rust functions."""
    return _thread_pool.get().pool


def _get_n_threads():
    """Get the number of threads in the current threadpool."""
    return _thread_pool.get().n_threads
