r"""Estimation of the mutual information between a sensitive value protected by masking and leakages in terms of the mutual information between each share and its corresponding leakages 

The `mrs_gerber_lemma` function takes as input the mutual information for each share separately (eventually for multiple sensitive values) 
and outputs an upper upper bound on the mutual information between the sensitive value and the leakages. 

By default, it is assumed that the leakages are expressed in bits. 
Eventually, a specific base for the unit of information can be specified.

The derivation is based on Mrs Gerber's lemma.  
In particular, it assumes that the the shares are leaking separately which should be ensured by a proper implementation of the masking countermeasure.  

Examples
--------

>>> from scalib.postprocessing import mrs_gerber_lemma 
>>> import numpy as np
>>> # Mutual information on three shares 
>>> MI_shares = np.array([.1,.2,.5])
>>> # Derive an upper bound on the mutual information of the protected secret 
>>> MI_sensitive = mrs_gerber_lemma(MI_shares,group_order=2**8, base=2)

Reference
---------

.. currentmodule:: scalib.postprocessing.noise_amplification

.. autosummary::
    :toctree:
    :nosignatures:
    :recursive:

    mrs_gerber_lemma

Notes
-----
The upper bound is based on the article :footcite:p:`mgl`,

References
----------

.. footbibliography::
"""

__all__ = ["mrs_gerber_lemma"]
import numpy as np


def phi(x: float) -> float:
    r"""Compute the DFT of the binary entropy.

    Parameters
    ----------
    x : array_like, f64
        Array of floats assumed to belong to [0,1]

    Returns
    -------
    An array corresponding to the image of the input array by the DFT of the binary entropy (in nats)
    """
    x = np.asarray(x)

    # Deal with special case 'x=1' with where
    y = np.where(
        x == 1, np.log(2), ((1 - x) * np.log1p(-x) + (1 + x) * np.log1p(x)) / 2
    )
    return y


def phi_derivative(x: float) -> float:
    r"""Compute the derivative of the DFT of the binary entropy.

    Parameters
    ----------
    x : array_like, f64
        Array of floats assumed to belong to [0,1]

    Returns
    -------
    An array corresponding to the image of the input array by the derivative of the DFT of the binary entropy
    """
    x = np.asarray(x)
    return np.arctanh(x)


def phi_second_derivative(x: float) -> float:
    r"""Compute the second derivative of the DFT of the binary entropy.

    Parameters
    ----------
    x : array_like, f64
        Array of floats assumed to belong to [0,1]

    Returns
    -------
    An array corresponding to the image of the input array by the second derivative of the DFT of the binary entropy
    """
    x = np.asarray(x)
    return (1 - x**2) ** (-1)


def phi_inv(y: float, niter=3) -> float:
    r"""Compute the inverse of the DFT of the binary entropy via Halley's root finding algorithm

    Parameters
    ----------
    y : array_like, f64
        Array of floats to apply the inverse, the values should belong to the interval [0,\log 2] in nats

    niter: int
        Number of iteration used in Halley's method. By default niter=3.

    Returns
    -------
    An array corresponding to the image of the input array by the inverse of the DFT of the binary entropy.
    """

    y = np.asarray(y)

    # Initial Guess of the inverse
    # Based on https://math.stackexchange.com/questions/3454390/find-the-approximation-of-the-inverse-of-binary-entropy-function
    x = np.sqrt(1 - (1 - y / np.log(2)) ** (4 / 3))

    # Iterative Halley's method for root finding with cubic convergence rate
    # See https://en.wikipedia.org/wiki/Halley's_method
    for _ in range(niter):

        f = phi(x) - y
        df = phi_derivative(x)
        ddf = phi_second_derivative(x)

        x -= (f * df) / (df**2 - f * ddf / 2)

    # We deal with the special case 0 and log(2) with where
    return np.where(y > 0, np.where(y < np.log(2), x, 1), 0)


def mrs_gerber_lemma(MI_shares: float, group_order: int, base=2) -> float:
    r"""Upper bound the mutual information of a sensitive value in terms of the mutual information for each of its share.

    Parameters
    ----------
    MI_shares : array_like, f64
        Mutual information for each share. Array must be of shape ``(ns,nv)`` where
        ``ns`` is the number of shares, ``nv`` the number of sensitive values.
    group_order : int
        Order of the Abelian in which the sensitive values are protected by masking.
    base : array_like, f64
        The base of information used, by default the information is in bits i.e. base=2.

    Returns
    -------
    Upper bound on the mutual information for all nv sensitive values based on Mrs Gerber's Lemma
    """

    MI_shares = np.asarray(MI_shares)

    # Check if the group order is a power of 2 (i.e. its bit expression contains a single 1)
    if (group_order & (group_order - 1)) == 0:
        MI_share_bits = MI_shares * np.log2(base)
        k = np.min(np.floor(MI_share_bits), axis=0)
        cliped_MI_shares = np.clip(0, 1, MI_share_bits - k) * np.log(2)
        MI = k * np.log(2) + phi(np.prod(phi_inv(cliped_MI_shares), axis=0))
    # Otherwise, use a weaker MGL based on Pinsker/reverse Pinsker inequalities
    else:
        beta = group_order**2 * 4 ** (1 / group_order)

        # Detect which shares are bellow the noise amplification ratio
        bellow = 2 * MI_shares * np.log(base) < 1

        # Multipy only the terms bellow the noise amplification ratio
        product = np.prod(np.where(bellow, 2 * MI_shares * np.log(base), 1), axis=0) / 4

        # If there is no share bellow the noise amplification ratio we return the minimum instead
        min_shares = np.min(MI_shares * np.log(base), axis=0)

        # Depending on the case we return either the minimum or the product of shares bellow noise amplification
        P = np.where(~np.any(bellow, axis=0), min_shares, product)

        MI_1 = np.log(1 + beta * P)
        MI_2 = (1 / group_order + np.sqrt(P)) * np.log(1 + group_order * np.sqrt(P))
        min_MI = np.minimum(np.minimum(MI_1, MI_2), min_shares)

        # Depending on the case we return either the minimum or the amplification lemma
        MI = np.where(~np.any(bellow, axis=0), min_shares, min_MI)

    # Conversion from nats to base 'base'
    MI /= np.log(base)
    return MI
