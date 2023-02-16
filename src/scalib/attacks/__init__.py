r"""
Soft Analytical Side-Channel Attack (SASCA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: scalib.attacks

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   FactorGraph
   BPState


Deprecated
~~~~~~~~~~

.. currentmodule:: scalib.attacks

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   SASCAGraph
"""

__all__ = ["FactorGraph", "BPState"]

from .sascagraph import SASCAGraph
from .factor_graph import FactorGraph, BPState
