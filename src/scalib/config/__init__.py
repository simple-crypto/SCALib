"""SCALib runtime configuration.

The configuration is fetched from different sources, with the following priority.
The highest-priority is the local configuration enabled with the `Config`
context manager (which can be nested, and follows thread/async context).
If no such local configuration is activated, the default configuration is used.
This configuration has default startup values (which may depend on environment
variables) that can be changed with the `default_config` function.

Configurable Behaviors
----------------------

Progress bars
^^^^^^^^^^^^^

By default, a progress bar is shown for all SCALlib operations if the output
(stderr) is a terminal.
The printing of progress bars can be disabled with the `show_progress` argument of `Config`.

Thread pools
^^^^^^^^^^^^

All computationally-heavy operations of SCALib are run on a thread pool.
Multiple thread pools can be used simultaneously, although using only the
default thread pool should be enough for most applications.

The default thread pool can be configured through the default `Config` or using
value of the environment variable `SCALIB_NUM_THREADS`, e.g.:

.. code-block::

    SCALIB_NUM_THREADS=8 python3 XXX.py

If `SCALIB_NUM_THREADS` is not set, reasonable default for the typical use of
SCALib is taken (it is currently the one given `here
<https://docs.rs/num_cpus/1.13.1/num_cpus/fn.get_physical.html>`_), but might
not be optimal for your workload.

Thread pools can be used locally in place of the default thread pool by
enabling a local config using `Config` as a context manager.
When initializing a `Config`, a `ThreadPool` can be provided through the
`threadpool` argument, or a new `ThreadPool` is created if the `n_threads`
argument is provided.
If neither of these arguements is provided, the currently active thread pool is
used.

Example
--------

>>> from scalib.config import default_config, Config
>>> # Set the default ThreadPool to 10 threads and do not show progress bars.
>>> default_config(n_threads=10, show_progress=False)
>>> # As an exception, the following computations run on 5 threads, with progress bars.
>>> # (NB: enabling progress is needed as Config inherits the current context by default.)
>>> with Config(n_threads=5, show_progress=False).activate():
...     # Do some computations with SCALib...
...     pass
>>> # One can also re-use a Config.
>>> # This is convenient and also allows sending jobs from multiple python threads to a single thread pool.
>>> config = Config(n_threads=5)
>>> with config.activate():
...     pass
>>> with config.activate():
...     pass

Reference
---------

.. currentmodule:: scalib.config

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   Config
   default_config
   ThreadPool
"""

__all__ = ["default_config", "Config", "ThreadPool"]

import contextvars
import contextlib
from typing import Optional

from scalib import _scalib_ext

from .threading import ThreadPool, _default_num_threads


_current = contextvars.ContextVar("scalib config")


class Config:
    """SCALib configuration.

    Configuration applicable to all SCALib computations.
    """

    def __init__(
        self,
        *,
        threadpool: Optional[ThreadPool] = None,
        n_threads: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ):
        self.inner = _scalib_ext.Config()
        if show_progress is not None:
            self.inner.show_progress(show_progress)
        else:
            self.inner.show_progress(get_config().inner.progress())
        if threadpool is not None:
            self.threadpool = threadpool
        elif n_threads is not None:
            self.threadpool = ThreadPool(n_threads)
        else:
            self.threadpool = get_config().threadpool

    @contextlib.contextmanager
    def activate(self):
        """Locally activate a Config."""
        restore_token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(restore_token)


_default = Config(n_threads=_default_num_threads(), show_progress=True)
_current.set(_default)


def default_config(**kwargs):
    """Configure the default Config.

    Arguments are the same as Config.
    """
    # Use __init__ function instead of creating new object as we want to update
    # the object in-place.
    _default.__init__(**kwargs)


def get_config():
    """Get the current config, to give to Rust functions."""
    return _current.get()
