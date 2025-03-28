r"""Estimation of the rank of a true key given likelihood of sub-keys.

Rank estimation estimates the rank of the full (e.g. 128-bit) key based on a
score for each value of its (e.g 8-bit) sub-keys. The scores must be additive
and positive, with lower score meaning lower rank (e.g. negative
log-likelihoods).

The `rank_accuracy` function allows to specify the desired precision of the
bound in bits. It is the high-level, and its usage is recommended over the
lower-level `rank_nbin` function.
That function gives direct, lower-level control to the core algorithm, allowing
to specify the number of bins in the histograms (whereas `rank_accuracy` tunes
this parameter automatically).

By default, the `rank_accuracy` function uses scaled histograms for convolution. 
Unlike the standard histogram approach, this method scales down the bins before
each convolution by a factor chosen to prevent exceeding a predefined limit.
After the convolution, the histogram is scaled back to restore its original magnitude.
This ensures correct functionality by avoiding negative values due to f64 overflows.
Additionally, two histograms are used — one for the lower bound and one for the upper
bound — to mitigate rounding errors and maintain precision.

Examples
--------

>>> from scalib.postprocessing import rank_accuracy
>>> import numpy as np
>>> # define the correct key
>>> key = np.random.randint(0,256,16)
>>> # Derive the score for each key byte
>>> scores = np.ones((16,256)) * 1E-5
>>> scores[np.arange(16),key] = 1
>>> # Compute the full key rank (correct key must have rank 1).
>>> (rmin,r,rmax) = rank_accuracy(-np.log(scores),key)
>>> assert r == 1


Reference
---------

.. currentmodule:: scalib.postprocessing.rankestimation

.. autosummary::
    :toctree:
    :nosignatures:
    :recursive:

    rank_accuracy
    rank_nbin

Notes
-----
The rank estimation algorithm is based on :footcite:p:`rankest_histograms`,
with the following
optimization: computation of histogram bins with higher score than the expected
key is skipped, since it has no impact on the final rank.

References
----------

.. footbibliography::
"""

__all__ = ["rank_nbin", "rank_accuracy", "HAS_NTL"]
import math

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.utils

HAS_NTL: bool = _scalib_ext.HAS_NTL


def rank_nbin(costs, key, nbins, method="hist"):
    r"""Estimate the rank of the full keys based on scores based on histograms.

    Warning: this is a low-level function, you probably want to use
    `rank_accuracy` instead.

    Parameters
    ----------
    costs : array_like, f64
        Cost for each of the sub-keys. Array must be of shape ``(ns,nc)`` where
        ``ns`` is the number of sub-keys, ``nc`` the possible values of each
        sub-keys.
    key : array_like, int
        Correct full key split in sub-keys. Array must be of shape ``(ns,)``.
    nbins : int
        Number of bins for each of the distributions.
    method : string
        Method used to estimate the rank. Can be the following:

        * "scaledhist": using two scaled histograms (default).
        * "hist": using histograms.
        * "naive": enumerate possible keys (very slow).
        * "histbignum": using NTL library, allows better precision.

    Returns
    -------
    (rmin, r, rmax): (float, float, float)

            - **rmin** is a lower bound for the key rank.
            - **r** is the estimated key rank.
            - **rmax** is an upper bound for the key rank.
    """
    with scalib.utils.interruptible():
        return _scalib_ext.rank_nbin(
            costs, key, nbins, _choose_merge_value(costs), method, get_config()
        )


def rank_accuracy(costs, key, acc_bit=1.0, method="scaledhist", max_nb_bin=2**26):
    r"""Estimate the rank of the full keys based on scores based on histograms.

    Parameters
    ----------
    costs : array_like, f64
        Cost for each of the sub-keys. Array must be of shape ``(ns,nc)`` where
        ``ns`` is the number of sub-keys, ``nc`` the possible values of each
        sub-keys.
    key : array_like, int
        Correct full key split in sub-keys. Array must be of shape ``(ns,)``.
    acc_bit : f64, default: 1.0
        Expected log2 accuracy for the key rank estimation.

        The algorithms attempts to get a result such that `rmax/rmin < 2^acc_bit`,
        but may not achieve this if the result is computationally intractable.
        In such a case, rank estimate and rank bounds are still returned, but
        the inequality may be violated.
    method : string
        Method used to estimate the rank. Can be the following:

        * "scaledhist": using two scaled histograms (default).
        * "hist": using histograms.
        * "ntl": using NTL library, allows better precision.
    max_nb_bin : int, default: :math:`2^{26}`
        Maximum number of bins to use.
        This fixes an upper bound on the computational cost of the algorithm
        (if too low, the requested accuracy might not be reached).

    Returns
    -------
    (rmin, r, rmax): (float, float, float)

            - **rmin** is a lower bound for the key rank.
            - **r** is the estimated key rank.
            - **rmax** is an upper bound for the key rank.
    """
    with scalib.utils.interruptible():
        return _scalib_ext.rank_accuracy(
            costs,
            key,
            2.0**acc_bit,
            _choose_merge_value(costs),
            method,
            max_nb_bin,
            get_config(),
        )


def _choose_merge_value(costs):
    """The merge parameter is the number of sub-keys to merge in a
    brute-force manner before computing histograms. Merging may improve
    accuracy at the expense of running time.
    Here we limit sub-histograms to :math:`2^{16}` values.
    """
    return min(len(costs), max(1, int(16 / math.log2(max(len(c) for c in costs)))))
