r"""Estimation of the mutual information between a sensitive value protected by masking and leakages in terms of the mutual information between each share and its corresponding leakages 

This function is usefull in the following setting.
You know that a sensitive value X is protected by masking so that it is shared into (S_0,...,S_d).  
You can observe leakages Y_0,...,Y_d for each corresponding shares.  
You have have estimated the leakages on each share via the mutual information I(S_i;Y_i).
Then, the mgl function provides an upper bound on the mutual information I(X; Y_0,...,Y_d).  
The obtained upper bound can then be used with other functions that provides security guarantees (such as success rate of an attack) in terms of mutual information. 

The `mgl` function takes as input the mutual information for each share separately (eventually for multiple sensitive values) 
and outputs an upper upper bound on the mutual information between the sensitive value and the leakages. 

By default, it is assumed that the leakages are expressed in bits. 
Eventually, a specific base for the unit of information can be specified.

The derivation is based on Mrs Gerber's lemma.  
In particular, it assumes that the the shares are leaking separately which should be ensured by a proper implementation of the masking countermeasure.  

Examples
--------

>>> from scalib.postprocessing import mgl 
>>> import numpy as np
>>> # Mutual information on three shares 
>>> mi_shares = np.array([.1,.2,.5])
>>> # Derive an upper bound on the mutual information of the protected secret 
>>> mi_sensitive = mgl(mi_shares,group_order=2**8, base=2)

Reference
---------

.. currentmodule:: scalib.postprocessing.noise_amplification

.. autosummary::
    :toctree:
    :nosignatures:
    :recursive:

    mgl

Notes
-----
The upper bound is based on the article :footcite:p:`mgl`,

References
----------

.. footbibliography::
"""

__all__ = ["mgl"]
import numpy as np
import numpy.typing as npt


def phi(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the DFT of the binary entropy.

    See equation 51 (Theorem 1) in the book chapter:
    Olivier Rioul, Julien Béguinot. The role of Mrs. Gerber’s Lemma for evaluating the information
    leakage of secret sharing schemes. Ioannis Kontoyiannis, Jason Klusowski, Cynthia Rush. Information
    Theory, Probability and Statistical Learning: A Festschrift in Honor of Andrew Barron, Springer, 2025.

    Parameters
    ----------
    x : array_like, f64
        Array of floats assumed to belong to [0,1]

    Returns
    -------
    An array corresponding to the image of the input array by the DFT of the binary entropy (in nats)
    """
    # Deal with special case 'x=1' with where
    y = np.where(
        x == 1, np.log(2), ((1 - x) * np.log1p(-x) + (1 + x) * np.log1p(x)) / 2
    )
    return y


def phi_derivative(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the derivative of the DFT of the binary entropy.

    Parameters
    ----------
    x : array_like, f64
        Array of floats assumed to belong to [0,1]

    Returns
    -------
    An array corresponding to the image of the input array by the derivative of the DFT of the binary entropy
    """
    return np.arctanh(x)


def phi_second_derivative(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the second derivative of the DFT of the binary entropy.

    Parameters
    ----------
    x : array_like, f64
        Array of floats assumed to belong to [0,1]

    Returns
    -------
    An array corresponding to the image of the input array by the second derivative of the DFT of the binary entropy
    """
    return (1 - x**2) ** (-1)


def phi_inv(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

    # We deal with the special case 0 and log(2) with masks
    inverse = np.empty_like(y)

    inverse[y == np.log(2)] = 1
    inverse[y == 0] = 0

    # We now compute the inverse for interior points
    interior = (y > 0) & (y < np.log(2))

    # Initial Guess of the inverse
    # Based on https://math.stackexchange.com/questions/3454390/find-the-approximation-of-the-inverse-of-binary-entropy-function
    x = np.sqrt(1 - (1 - y[interior] / np.log(2)) ** (4 / 3))

    # Iterative Halley's method for root finding with cubic convergence rate
    # See https://en.wikipedia.org/wiki/Halley's_method
    for _ in range(3):

        f = phi(x) - y[interior]
        df = phi_derivative(x)
        ddf = phi_second_derivative(x)

        x -= (f * df) / (df**2 - f * ddf / 2)

    inverse[interior] = x

    return inverse


def mgl(mi_shares: npt.ArrayLike, group_order: int, base=2) -> npt.NDArray[np.float64]:
    r"""Upper bound the mutual information of a sensitive value in terms of the mutual information for each of its share.


    Parameters
    ----------
    mi_shares : array_like, f64
        Mutual information for each share. Array must be of shape ``(ns,nv)`` where
        ``n_shares`` is the number of shares, ``nv`` the number of sensitive values.
    group_order : int
        Order of the group in which the sensitive values are protected by masking.
    base : array_like, f64
        The base of information used, by default the information is in bits i.e. base=2.

    Returns
    -------
    Upper bound on the mutual information for all nv sensitive values based on Mrs Gerber's Lemma.
    """

    mi_shares = np.asarray(mi_shares, dtype=np.float64)

    if not (mi_shares >= 0).all():
        raise ValueError(
            "Invalid inputs the mutual information on each share should be positive."
        )
    if not (mi_shares <= np.log(group_order) / np.log(base)).all():
        raise ValueError(
            "Invalid inputs the mutual information on each share should be less than the logarithm in base base of the group order."
        )

    # Check if the group order is a power of 2 (i.e. its bit expression contains a single 1)
    # See Theorem 1  in the book chapter tosee the mgl reformulated using DFT as implemented here:
    # Olivier Rioul, Julien Béguinot. The role of Mrs. Gerber’s Lemma for evaluating the information
    # leakage of secret sharing schemes. Ioannis Kontoyiannis, Jason Klusowski, Cynthia Rush. Information
    # Theory, Probability and Statistical Learning: A Festschrift in Honor of Andrew Barron, Springer, 2025.
    if (group_order & (group_order - 1)) == 0:
        mi_share_bits = mi_shares * np.log2(base)
        k = np.min(np.floor(mi_share_bits), axis=0)
        clipped_mi_shares = np.clip(0, 1, mi_share_bits - k) * np.log(2)
        mi = k * np.log(2) + phi(np.prod(phi_inv(clipped_mi_shares), axis=0))
    # Otherwise, use a weaker MGL based on Pinsker/reverse Pinsker inequalities
    else:
        beta = group_order**2 * 4 ** (1 / group_order)

        # Detect which shares are below the noise amplification ratio.
        below = 2 * mi_shares * np.log(base) < 1

        # Multipy only the terms bellow the noise amplification ratio
        product = np.prod(np.where(below, 2 * mi_shares * np.log(base), 1), axis=0) / 4

        # If there is no share bellow the noise amplification ratio we return the minimum instead
        min_shares = np.min(mi_shares * np.log(base), axis=0)

        # Depending on the case we return either the minimum or the product of shares below noise amplification
        P = np.where(~np.any(below, axis=0), min_shares, product)

        mi_1 = np.log(1 + beta * P)
        mi_2 = (1 / group_order + np.sqrt(P)) * np.log1p(group_order * np.sqrt(P))
        min_mi = np.minimum(np.minimum(mi_1, mi_2), min_shares)

        # Depending on the case we return either the minimum or the amplification lemma
        mi = np.where(~np.any(below, axis=0), min_shares, min_mi)

    # Conversion from nats to base 'base'
    mi /= np.log(base)
    return mi
