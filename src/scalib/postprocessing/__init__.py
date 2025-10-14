r"""
Rank Estimation
^^^^^^^^^^^^^^^

.. currentmodule:: scalib.postprocessing

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   scalib.postprocessing.rankestimation
   scalib.postprocessing.leakage_to_success
"""

__all__ = [
    "rankestimation",
    "success_rate",
    "guessing_entropy",
    "log_guessing_entropy",
    "median",
]

from .rankestimation import rank_nbin, rank_accuracy
from .leakage_to_success import (
    success_rate,
    guessing_entropy,
    log_guessing_entropy,
    median,
)
