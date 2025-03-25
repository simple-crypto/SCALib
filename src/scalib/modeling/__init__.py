r"""

Linear Discriminant Analysis (LDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LDA is a also known as pooled Gaussian templates in the side-channel litterature.
:class:`LDAClassifier` is the main implementation of the LDA, and
:class:`MultiLDA` is a convenience wrapper for profiling multiple
variables using the same traces (but possibly different sets of POIs).

.. currentmodule:: scalib.modeling

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   LDAClassifier
   MultiLDA
   LdaAcc
   Lda
   RLDAClassifier
"""

__all__ = [
    "LDAClassifier",
    "MultiLDA",
    "LdaAcc",
    "Lda",
    "RLDAClassifier",
]

from .lda import (
    LDAClassifier,
    MultiLDA,
    LdaAcc,
    Lda,
)
from .rldaclassifier import RLDAClassifier
