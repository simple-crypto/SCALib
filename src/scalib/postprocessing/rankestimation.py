r"""Estimation of the rank of a true key given likelihood of sub-keys.

Rank estimation estimates the rank of the full (e.g. 128-bit) key based on a
score for each value of its (e.g 8-bit) sub-keys. The scores must be additive
and positive (e.g. negative log probabilities).

`rank_accuracy` allows to specify the desired precision of the bound in
bits (may not be achieved if computationally untractable), while `rank_nbin`
gives direct, lower-level control over the number of bins in the historgrams.


Notes
-----
The rank estimation algorithm is based on [1]_, with the following
optimization: computation of histogram bins with higher score than the expected
key is skipped, since it has no impact on the final rank.

.. [1] "Simple Key Enumeration (and Rank Estimation) Using Histograms: An
   Integrated Approach", R. Poussier, F.-X. Standaert, V. Grosso in CHES2016.
"""

import math

from scalib import _scalib_ext


def rank_nbin(costs, key, nbins, method="hist"):
    r"""Estimate the rank of the full keys based on scores based on histograms.

    Parameters
    ----------
    costs : array_like, f64
        Cost for each of the sub-keys. Array must be of shape `(ns,nc)` where
        `ns` is the number of sub-keys, `nc` the possible values of each
        sub-keys.
    key : array_like, int
        Correct full key split in sub-keys. Array must be of shape `(ns,)`.
    nbins : int
        Number of bins for each of the distributions.
    method : string
        Method used to estimate the rank. Can be the following:

        * "hist": using histograms (default).
        * "ntl": using NTL library, allows better precision.

    Returns
    -------
    (rmin, r, rmax): (float, float, float)

            - **rmin** is a lower bound for the key rank.
            - **r** is the stimated key rank.
            - **rmax** is an upper bound for the key rank.
    """
    return _scalib_ext.rank_nbin(costs, key, nbins, _choose_merge_value(costs), method)


def rank_accuracy(costs, key, acc_bit=1.0, method="hist", max_nb_bin=2 ** 26):
    r"""Estimate the rank of the full keys based on scores based on histograms.

    Parameters
    ----------
    costs : array_like, f64
        Cost for each of the sub-keys. Array must be of shape `(ns,nc)` where
        `ns` is the number of sub-keys, `nc` the possible values of each
        sub-keys.
    key : array_like, int
        Correct full key split in sub-keys. Array must be of shape `(ns,)`.
    acc_bit : f64, default: 1.0
        Expected log2 accuracy for the key rank estimation. `acc_bit` must so be
        set to `log2(rank_max/rank_min)`.
    method : string
        Method used to estimate the rank. Can be the following:

        * "hist": using histograms (default).
        * "ntl": using NTL library, allows better precision.
    max_nb_bin : int, default: 2**26
        Maximum number of bins to use (if too low, the requested accuracy might
        not be reached).

    Returns
    -------
    (rmin, r, rmax): (float, float, float)

            - **rmin** is a lower bound for the key rank.
            - **r** is the stimated key rank.
            - **rmax** is an upper bound for the key rank.
    """

    return _scalib_ext.rank_accuracy(
        costs, key, 2.0 ** acc_bit, _choose_merge_value(costs), method, max_nb_bin
    )


def _choose_merge_value(costs):
    """The merge parameter is the number of sub-keys to merge in a
    brute-force manner before computing histograms. Merging may improve
    accuracy at the expense of running time.
    Here we limit sub-histograms to 2**16 values.
    """
    return min(len(costs), max(1, int(16 / math.log2(max(len(c) for c in costs)))))
