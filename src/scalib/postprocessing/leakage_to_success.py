r"""Lower bounds on the log-guessing entropy, log of the guessing entropy, log of the median rank and upper bound on the proability of a successful attack in presence of key enumeration.

The bounds depends on the leakage as measured by mutual information, the size of the secret key (number of bits) and eventually the number of key enumerated.

Let say that an adversary acquires side channel information about a uniformly distributed secret K  through some side channel information Y.
At the end of a side channel attack the adversary produces a ranking of the different key hypotheses from most likely to least likely. 
We let :math:`\mathrm{rank}(K|Y)` be the position of the correct key in this ranking. 
The probability distribution of :math:`\mathrm{rank}(K|Y)` provides securitry guarantees about the performances of the attacker.  
It is hard to obtain :math:`\mathrm{rank}(K|Y)` for a full key, hoverwer, it is often possible to upper bound the informational leakages :math:`I(K;Y)`.
Using :math:`I(K;Y)` we provide bounds on some statistics of the rank:

* The probability of success of level L : 

.. math::
    \mathbb{P}_{s,L}(K|Y) = \mathbb{P}( \mathrm{rank}(K|Y) \leq L), 

which bounds the probability of a successful attack when L keys are enumerated; 

* The guessing entropy : 

.. math::
    G(K|Y) = \mathbb{E}[\mathrm{rank}(K|Y)], 

which corresponds to the average enumeration time to recover the secret key; 

* The median rank, which is the minimum number of hypotheses to enumerate to achieve a success rate of 1/2 i.e. med is the mimimum value such that 

.. math::
    \mathrm{Med}(K|Y) = \min \{ m \mid \mathbb{P}( \mathrm{rank}(K|Y) \leq m) \geq 1/2 \}; 

* The log-guessing entropy,

.. math:: 
    \mathrm{LG}(K|Y) = \mathbb{E}[ \log_2 \mathrm{rank}(K|Y)],

which is corresponds to the average security parameter of the implementation (it is more conservative than the guessing entropy).


Reference
---------
.. currentmodule:: scalib.postprocessing.leakage_to_success

.. autosummary::
    :toctree:
    :nosignatures:
    :recursive:

    success_rate
    guessing_entropy
    log_guessing_entropy
    median

Notes
-----
The upper bound is based on the article :footcite:p:`figMerit`,
References
----------
.. footbibliography::
"""

__all__ = ["success_rate", "guessing_entropy", "log_guessing_entropy", "median"]
import numpy as np
from scipy.special import gamma, bernoulli, factorial
from scipy.optimize import brentq


def binary_divergence_log(log_p: float, log_q: float) -> float:
    r"""Outputs the binary divergence d(p||q) in a numerically stable way when log(p) and log(q) are given.

    Parameters
    ----------
    log_p, log_q : float

    Returns
    -------
    An estimation of the binary divergence d(p||q).
    """
    if log_p == 0:
        return -log_q

    p = np.exp(log_p)
    q = np.exp(log_q)

    return p * (log_p - log_q) - np.expm1(log_p) * (np.log1p(-p) - np.log1p(-q))


def success_rate(
    mutual_information: np.ndarray[float],
    key_size: int,
    enumeration_effort: float,
    base=2,
) -> np.ndarray[float]:
    r"""Output an upper bound on the logarithm in base 2 of the probability of correctly guessing a secret key of of size key_size bits
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary and the adversary is allowed to enumerate up to
    :math:`L = 2^{\mathrm{enumeration \; effort}}` key hypotheses.

    .. math::
        \log_2 \mathbb{P}_{s,L}(K|Y) = \log_2 \mathbb{P}( \mathrm{rank}(K|Y) \leq L).

    Parameters
    ----------
    mutual_information : array_like, f64
        Upper bound on the mutual information expressed in base 'base'
    key_size : int
        Number of bits of the secrets
    enumeration_effort : array_like, f64
        logarithm in base 2 of the number of key hypothesis that the adversary is allowed to enumerate
    base : f64
        Base for the information used (by default base=2 i.e. the base is the bit)
    niter : int
        Number of iteration of the dichotomic search used internaly. A higher niter improves the precision
        but also the runtime and vice versa.  By default it is set to 30 iterations.

    Returns
    -------
    Output an upper bound on the logarithm in base 'base' of the probability of correctly guessing a secret key of of size key_size bits when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary and the adversary is allowed to enumerate up to 2^enumeration_effort key_hypotheses.


    Examples
    --------
    >>> from scalib.postprocessing import success_rate
    >>> import numpy as np
    >>> key_size=128
    >>> enumeration_effort=32
    >>> mi = np.random.uniform(low=0, high=key_size, size=1)
    >>> sr = success_rate(mi,key_size,enumeration_effort)
    """
    if not (mutual_information >= 0).all():
        raise ValueError("Invalid inputs, the mutual information should be positive.")
    if not (mutual_information * np.log2(base) <= key_size).all():
        raise ValueError(
            "Invalid inputs, the mutual information in bits should be less than the key_size."
        )

    mutual_information = np.atleast_1d(mutual_information)
    mi_nats = mutual_information * np.log(base)
    log_q = (enumeration_effort - key_size) * np.log(2)  # in nats

    log_sr = np.empty_like(mutual_information)
    for i in range(len(mutual_information)):
        if mi_nats[i] >= -log_q:
            log_sr[i] = 0
        else:
            log_sr[i] = brentq(
                lambda log_p: binary_divergence_log(log_p, log_q) - mi_nats[i],
                0,
                log_q,
                xtol=10**-2,
                maxiter=100,
                full_output=False,
            )

    return log_sr / np.log(2)  # in bits


def massey_inequality(x: float) -> float:
    r"""
    This sub-function implements the optimal version of Massey's inequality.

    Massey's inequality when optimaly writen reads as  H(X|Y) >= G(X|Y) Hb( G(X|Y)^{-1}) where Hb is the binary entropy.

    Let x = log( G(X|Y) ) this reads as  H(X|Y) >= x - (exp(x)-1) * log(1 - exp(-x)).

    We implement this function using exp1m and log1p to be stable.
    """
    if x == 0:
        return 0
    return x - np.expm1(x) * np.log1p(-np.exp(-x))


def massey_inequality_inverse(y: float) -> float:
    r"""
    This sub-function implements the inverse of the optimal Massey's inequality using brent's rootfinding algorithm.
    """
    # Ensure that y is an array
    y = np.atleast_1d(y)

    inverse = np.zeros_like(y)
    for i in range(len(y)):
        # A precise approximation of the inverse would be y  - 1 + log(1 + exp(1-y[i])/2)
        lb = np.maximum(0, y[i] - 1)
        ub = y[i]
        inverse[i] = brentq(
            lambda x: massey_inequality(x) - y[i],
            lb,
            ub,
            xtol=10**-5,
            maxiter=100,
            full_output=False,
        )
    return inverse


def guessing_entropy(
    mutual_information: np.ndarray[float], key_size: int, base=2
) -> np.ndarray[float]:
    r"""Output a lower bound on the logarithm in base 2 of the guessing entropy
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary:

    .. math::
        \mathrm{G}(K|Y) = \log_2 \mathbb{E}[ \mathrm{rank}(K|Y)].

    Parameters
    ----------
    mutual_information : array_like, f64
        Upper bound on the mutual information expressed in base 'base'
    key_size : int
        Number of bits of the secrets
    base : f64
        Unit for the logarithms used (by default base=2 i.e. the unit is the bit)

    Returns
    -------
    Lower bound on the logarithm in  base 'base' of the guessing entropy

    Examples
    --------
    >>> from scalib.postprocessing import guessing_entropy
    >>> import numpy as np
    >>> key_size=128
    >>> mi = np.random.uniform(low=0, high=key_size, size=1)
    >>> ge = guessing_entropy(mi, key_size)
    """
    if not (mutual_information >= 0).all():
        raise ValueError("Invalid inputs, the mutual information should be positive.")
    if not (mutual_information * np.log2(base) <= key_size).all():
        raise ValueError(
            "Invalid inputs, the mutual information in bits should be less than the key_size."
        )

    MI_nats = mutual_information * np.log(base)
    key_size_nats = key_size * np.log(2)
    entropy_nats = key_size_nats - MI_nats

    finite_size_bound = np.zeros_like(mutual_information)
    sqrt = np.sqrt(np.tanh(key_size_nats / 2) * 2 * MI_nats / 3)
    mask = sqrt < 1
    finite_size_bound[mask] = (
        key_size_nats
        - np.log(2)
        + np.log1p(np.exp(-key_size_nats))
        + np.log1p(-sqrt[mask])
    )
    generic_bound = massey_inequality_inverse(entropy_nats)
    bound = np.max([finite_size_bound, generic_bound], axis=0) / np.log(2)  # in bits
    return bound


def partial_sum_zeta(a: np.ndarray[float], p: int, q: int) -> np.ndarray[float]:
    r"""Output the partial sum sum_{i=p}^q i^{-a}

    Parameters
    ----------
    a : array_like, f64
        Evaluation points
    p,q : int
        Summation boundaries
    Returns
    -------
    The sum sum_{i=p}^q i^{-a}
    """
    I = np.arange(p, q + 1).reshape((-1, 1)).astype(float)
    return np.sum(I ** -np.expand_dims(a, axis=0), axis=0)


def euler_maclaurin_correction(
    a: np.ndarray[float], k: int, log_M: float, order: int
) -> np.ndarray[float]:
    r"""This sub-function computes the first "order" Euler-MacLaurin corrective terms needed to approximate the sum
    sum_{i=k}^{exp(log_M)} i^{-a}.

    These terms depends on both the Bernoulli numbers and the evalutation of the derivatives of i -> i^{-a}.
    See  https://en.wikipedia.org/wiki/Euler%E2%80%93Maclaurin_formula

    """
    if order == 0:
        return np.zeros_like(a)
    d_O = (np.arange(1, order + 1) * 2 - 1).reshape((order,) + (1,) * a.ndim)
    derivatives_kp1 = -((k + 1) ** (-a - d_O)) * gamma(a + d_O) / gamma(a)
    derivatives_M = -np.exp2(log_M * (-a - d_O)) * gamma(a + d_O) / gamma(a)
    EM_coeff = (
        bernoulli(2 * order)[2::2] / factorial(np.arange(1, order + 1) * 2)
    ).reshape((order,) + (1,) * a.ndim)
    return np.sum((derivatives_M - derivatives_kp1) * EM_coeff, axis=0)


def log_zeta_larger_than_1(
    a: np.ndarray[float], log_M: float, order: int, k: int
) -> np.ndarray[float]:
    r"""This sub-function approximates the log of the truncated zeta function log(sum_{i=1}^{exp(log_M)} i^{-a})
    for a parameter a > 1.
    """
    # Partial Sum
    partial_sum = partial_sum_zeta(a, 1, k)

    # Boundary coerrections
    correction = ((k + 1) ** (-a) + np.exp2(-a * log_M)) / 2

    # Integral term
    integral = (np.exp2((1 - a) * log_M) - (k + 1) ** (1 - a)) / (1 - a)

    # Euler Maclaurin correction
    correction += euler_maclaurin_correction(a, k, log_M, order)

    return np.log(partial_sum + integral + correction)


def log_zeta_less_than_1(
    a: np.ndarray[float], log_M: float, order: int, k: int
) -> np.ndarray[float]:
    r"""This sub-function approximates the log of the truncated zeta function log(sum_{i=1}^{exp(log_M)} i^{-a})
    for a parameter 0 < a < 1.
    """
    log_M_nats = log_M * np.log(2)
    partial_sum = partial_sum_zeta(a, 1, k)
    correction = ((k + 1) ** (-a) + np.exp2(-a * log_M)) / 2

    # Euler Maclaurin correction
    correction += euler_maclaurin_correction(a, k, log_M, order)

    # return np.log(partial_sum + integral + correction)
    return (
        (1 - a) * log_M_nats
        - np.log1p(-a)
        + np.log1p(
            (1 - a) * (partial_sum + correction) * np.exp2((a - 1) * log_M)
            - np.exp2((a - 1) * (log_M - np.log2(k + 1)))
        )
    )


def log_zeta(
    a: np.ndarray[float], log_M: float, order: int = 10, k: int = 50
) -> np.ndarray[float]:
    r"""This sub-function approximates the log of the truncated zeta function log(sum_{i=1}^{exp(log_M)} i^{-a}).
    Depending on wether 0<a<1, a == 1 or a>1 it uses a specific expression taylored to ensure numerical stability.
    """
    # Ensure that a is treated as a Numpy Array
    a = np.asarray(a)

    # If less than 2 ** 10 terms we compute the exact sum
    if log_M < 10:
        return np.log(partial_sum_zeta(a, 1, int(np.exp2(log_M))))

    # Otherwise, we estimate numerically the sum via:
    #  - exact sum of the first term k terms sum_{i=1}^k i^{-a}
    #  - comparison to an integral for the "sum tail" sum_{i=k+1}^M i^{-a}
    #  - eventually corrective terms as given by Euler MacLaurin summation formula

    # Allocate array to store the result
    log_zeta_a = np.empty_like(a, dtype=float)

    a_less_than_1 = a < 1
    a_equal_1 = a == 1
    a_larger_than_1 = a > 1

    if a_equal_1.any():
        log_M_nats = log_M * np.log(2)
        log_zeta_a[a_equal_1] = np.log(
            log_M_nats + np.euler_gamma + np.log1p(np.exp(-log_M_nats) / 2)
        )
    if a_less_than_1.any():
        log_zeta_a[a_less_than_1] = log_zeta_less_than_1(
            a[a_less_than_1], log_M, order, k
        )
    if a_larger_than_1.any():
        log_zeta_a[a_larger_than_1] = log_zeta_larger_than_1(
            a[a_larger_than_1], log_M, order, k
        )

    return log_zeta_a


def log_guessing_entropy(
    mutual_information: np.ndarray[float],
    key_size: int,
    base: int = 2,
    order: int = 10,
    k: int = 50,
) -> np.ndarray[float]:
    r"""Output a lower bound on the log-guessing entropy in base 2 of the rank of the key of size key_size
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary:

    .. math::
        \mathrm{LG}(K|Y) = \mathbb{E}[\log_2 \mathrm{rank}(K|Y)].

    Parameters
    ----------
    mutual_information : array_like, f64
        Upper bound on the mutual information expressed in base 'base'
    key_size : int
        Number of bits of the secrets
    base : f64
        Base for the information used (by default base=2 i.e. the base is the bit)
    order : int
        Number of Euler-Maclaurin corrective term used in the evaluation (by default order=10)
    k : int
        Number of term in the sum that we compute exactly (by default k=50)
    Returns
    -------
    Lower bound on the logarithm in  base 'base' of the log-guessing entropy rank of the key of size key_size when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary.

    Examples
    --------
    >>> from scalib.postprocessing import log_guessing_entropy
    >>> import numpy as np
    >>> key_size=128
    >>> mi = np.random.uniform(low=0, high=key_size, size=1)
    >>> lg = log_guessing_entropy(mi,key_size)
    """
    if not (mutual_information >= 0).all():
        raise ValueError("Invalid inputs, the mutual information should be positive.")
    if not (mutual_information * np.log2(base) <= key_size).all():
        raise ValueError(
            "Invalid inputs, the mutual information in bits should be less than the key_size."
        )

    mutual_information = np.atleast_1d(mutual_information)

    MI_nats = mutual_information * np.log(base)
    key_size_nats = key_size * np.log(2)
    entropy_nats = key_size_nats - MI_nats

    A = np.hstack(
        [np.geomspace(10**-10, 1, 90), 1, np.linspace(1 + 10**-10, 5, 10)]
    ).reshape((-1,))
    LG = np.max(
        (np.expand_dims(entropy_nats, axis=1) - log_zeta(A, key_size, order, k)) / A,
        axis=1,
    )
    return np.maximum(0, LG / np.log(2))  # in bits


def median(
    mutual_information: np.ndarray[float], key_size: int, base: float = 2
) -> np.ndarray[float]:
    r"""Output a lower bound on the logarithm in base 2 of the median rank of the key of size key_size
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary:

    .. math::
        \log_2 \mathrm{Med}(K|Y) = \log_2 \min \{ m \mid \mathbb{P}( \mathrm{rank}(K|Y) \leq m) \geq 1/2 \}.

    Parameters
    ----------
    mutual_information : array_like, f64
        Upper bound on the mutual information expressed in base 'base'
    key_size : int
        Number of bits of the secrets
    base : f64
        Base for the information used (by default base=2 i.e. the base is the bit)

    Returns
    -------
    Lower bound on the logarithm in  base 'base' of the median rank of the key of size key_size when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary.

    Examples
    --------
    >>> from scalib.postprocessing import median
    >>> import numpy as np
    >>> key_size=128
    >>> mi = np.random.uniform(low=0, high=key_size, size=10**3)
    >>> med = median(mi,key_size)
    """
    if not (mutual_information >= 0).all():
        raise ValueError("Invalid inputs, the mutual information should be positive.")
    if not (mutual_information * np.log2(base) <= key_size).all():
        raise ValueError(
            "Invalid inputs, the mutual information in bits should be less than the key_size."
        )

    med = np.zeros_like(mutual_information)
    # We use log1p and exp1m to have a stable expression
    # We deal with eventual overflows in the exponential for large mutual information next
    sqrt = np.sqrt(-np.expm1(-2 * mutual_information * np.log(base)))
    med[sqrt < 1] = key_size - 1 + np.log1p(-sqrt[sqrt < 1]) / np.log(2)

    # To avoid numerical overflows we use an assymptotic expansion for large values of the mutual information
    # sqrt = 1 means that 1-exp(-2 * mutual_information * np.log(base)) is numerically simplified to 1 (large value of mutual_information)
    # Hence 1-sqrt is zero and the log overflows.
    # In this case we use the Taylor expansion log(1-sqrt(1-exp(-x))) = - (log 2) - x
    med[sqrt == 1] = np.maximum(
        0, key_size - 2 - 2 * mutual_information[sqrt == 1] * np.log2(base)
    )

    return med
