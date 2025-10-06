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

__all__ = ["rankestimation", "mrs_gerber_lemma"]

from .rankestimation import rank_nbin, rank_accuracy
from .noise_amplification import mrs_gerber_lemma
