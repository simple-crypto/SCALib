r"""
This module contains several well-know side-channel metrics.

.. currentmodule:: scalib.metrics

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   SNR
   ttest
   RLDAInformationEstimator
"""

__all__ = ["SNR", "ttest", "RLDAInformationEstimator"]

from .snr import SNR
from .ttest import Ttest
from .ttest import MTtest
from .information import RLDAInformationEstimator
