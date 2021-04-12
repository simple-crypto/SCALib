r"""This modules contains functions in order to estimate the rank of the full
key based on a list of score for each of its sub-keys. The scores are expected
to be the negative log probabilities.


Notes
-----
[1] "Simple Key Enumeration (and Rank Estimation) Using Histograms: An
Integrated Approach", R. Poussier, F.-X. Standaert, V. Grosso in CHES2016.
"""

import math

from scaffe import _scaffe_ext


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
    rmin : f64
        Lower bound for the rank key.
    r : f64
        Estimated key rank.
    rmax : f64
        Upper bound for the rank key.

    """
    return _scaffe_ext.rank_nbin(costs, key, nbins, choose_merge_value(costs), method)


def rank_accuracy(costs, key, acc_bit=1.0, method="hist"):
    r"""Estimate the rank of the full keys based on scores based on histograms.

    Parameters
    ----------
    costs : array_like, f64
        Cost for each of the sub-keys. Array must be of shape `(ns,nc)` where
        `ns` is the number of sub-keys, `nc` the possible values of each
        sub-keys.
    key : array_like, int
        Correct full key split in sub-keys. Array must be of shape `(ns,)`.
    acc_bit : f64
        Expected log2 accuracy for the key rank estimation. `acc_bit` must so be
        set to `log2(rank_max/rank_min)`.
    method : string
        Method used to estimate the rank. Can be the following:

        * "hist": using histograms (default).
        * "ntl": using NTL library, allows better precision.

    Returns
    -------
    rmin : f64
        Lower bound for the rank key.
    r : f64
        Estimated key rank.
    rmax : f64
        Upper bound for the rank key.
    """

    return _scaffe_ext.rank_accuracy(
        costs, key, 2.0 ** acc_bit, choose_merge_value(costs), method
    )


def choose_merge_value(costs):
    """The merge parameter is the number of sub-keys to merge in a
    brute-force manner before computing histograms. Merging may improve
    accuracy at the expense of running time.
    Here we limit sub-histograms to 2**16 values.
    """
    return max(1, int(16 / math.log2(max(len(c) for c in costs))))
