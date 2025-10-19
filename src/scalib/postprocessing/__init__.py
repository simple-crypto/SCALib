r"""
Rank Estimation
^^^^^^^^^^^^^^^

.. currentmodule:: scalib.postprocessing

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   scalib.postprocessing.rankestimation
   scalib.postprocessing.noise_amplification
"""

__all__ = ["rankestimation", "mgl"]

from .rankestimation import rank_nbin, rank_accuracy
from .noise_amplification import mgl
