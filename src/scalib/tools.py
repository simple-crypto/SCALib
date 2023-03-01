"""General non-SCA tools that come handy when using SCALib.

Reference
---------

.. currentmodule:: scalib.tools

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   ContextExecutor
"""

__all__ = ["ContextExecutor"]

from concurrent.futures import ThreadPoolExecutor
import contextvars


def restore_context(context):
    for var, value in context.items():
        var.set(value)


class ContextExecutor(ThreadPoolExecutor):
    """concurrent.futures.ThreadPoolExecutor with automatic propagation of contextvars.

    The context is captured at the creation of the executor.
    """

    def __init__(self, *args, **kwargs):
        context = contextvars.copy_context()
        super().__init__(*args, **kwargs, initializer=lambda: restore_context(context))
