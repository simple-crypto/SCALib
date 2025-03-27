r"""
Soft Analytical Side-Channel Attack (SASCA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: scalib.attacks

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   CPA
   FactorGraph
   GenFactor
   BPState
"""

__all__ = ["FactorGraph", "BPState", "GenFactor", "CPA"]

from .factor_graph import FactorGraph, BPState, GenFactor
from .cpa import CPA
