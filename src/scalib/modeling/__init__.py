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
"""

__all__ = ["LDAClassifier", "MultiLDA"]

from .ldaclassifier import LDAClassifier, MultiLDA
