r"""Lower bounds on the log-guessing entropy, log of the guessing entropy, log of the median rank and upper bound on the proability of a successful attack in presence of key enumeration.

The bounds depends on the leakage as measured by mutual information, the size of the secret key (number of bits) and eventually the number of key enumerated.

Examples
--------
>>> from scalib.postprocessing import success_rate, guessing_entropy, log_guessing_entropy, median
>>> import numpy as np
>>> key_size=128
>>> enumeration_effort=32
>>> mi = np.random.uniform(low=0, high=key_size, size=10**3)
>>> sr = success_rate(mi,key_size,enumeration_effort)
>>> med = median(mi,key_size)
>>> lg = log_guessing_entropy(mi,key_size)
>>> ge = guessing_entropy(mi, key_size)

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


def binary_divergence_log(
    log_p: np.ndarray[float], log_q: np.ndarray[float]
) -> np.ndarray[float]:
    p = np.exp(log_p)
    q = np.exp(log_q)
    return p * (log_p - log_q) + (1 - p) * (np.log1p(-p) - np.log1p(-q))


def success_rate(
    mutual_information: np.ndarray[float],
    key_size: int,
    enumeration_effort: float,
    base=2,
    niter=30,
):
    r"""Output an upper bound on the logarithm in base 'base' of the probability of correctly guessing a secret key of of size key_size bits
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary and the adversary is allowed to enumerate up to
    2 ** enumeration_effort key hypotheses.

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
    """

    MI_nats = mutual_information * np.log(base)
    log_q = (enumeration_effort - key_size) * np.log(2)  # in nats

    # Initial intervall for the dichotomic search
    log_p_lb = np.array(log_q).reshape((1,))
    log_p_ub = np.array(0).reshape((1,))

    # Dichotomic serach
    for _ in range(niter):
        log_p_mid = (log_p_lb + log_p_ub) / 2
        y = binary_divergence_log(log_p_mid, log_q) - MI_nats
        log_p_lb, log_p_ub = np.where(
            y > 0, (log_p_lb, log_p_mid), (log_p_mid, log_p_ub)
        )
    return log_p_ub / np.log(base)  # in base 'base'


def f(x):
    return np.where(x > 0, x - np.expm1(x) * np.log1p(-np.exp(-x)), 0)


def f_inv(y, niter=20):

    # Ensure that y is an array
    y = np.atleast_1d(y)

    # Initial guess
    x_lb = np.maximum(np.log(1), y - 1 + np.log1p(np.exp(1 - y) / 2)).reshape((-1,))
    x_ub = np.maximum(np.log(2), y - 1 + np.log1p(np.exp(1 - y) / 2 + 1)).reshape((-1,))

    assert (f(x_lb) <= y).all()
    assert (f(x_ub) >= y).all()

    # Dichotomic search
    for _ in range(niter):
        x_mid = (x_lb + x_ub) / 2
        x_lb, x_ub = np.where(f(x_mid) - y < 0, (x_mid, x_ub), (x_lb, x_mid))

    return x_lb


def guessing_entropy(mutual_information, key_size, base=2):
    r"""Output a lower bound on the logarithm in base 'base' of the guessing entropy
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary.

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
    """
    MI_nats = mutual_information * np.log(base)
    key_size_nats = key_size * np.log(2)
    entropy_nats = key_size_nats - MI_nats
    sqrt = np.sqrt(np.tanh(key_size_nats / 2) * 2 * MI_nats / 3)

    finite_size_bound = np.zeros_like(mutual_information)
    mask = sqrt < 1
    finite_size_bound[mask] = (
        key_size_nats
        - np.log(2)
        + np.log1p(np.exp(-key_size_nats))
        + np.log1p(-sqrt[mask])
    )
    generic_bound = f_inv(entropy_nats)
    bound = np.max([finite_size_bound, generic_bound], axis=0) / np.log(base)
    return bound


def partial_sum_zeta(a, p, q):
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


def euler_maclaurin_correction(a, k, log_M, order):
    if order == 0:
        return np.zeros_like(a)
    d_O = (np.arange(1, order + 1) * 2 - 1).reshape((order,) + (1,) * a.ndim)
    derivatives_kp1 = -((k + 1) ** (-a - d_O)) * gamma(a + d_O) / gamma(a)
    derivatives_M = -np.exp2(log_M * (-a - d_O)) * gamma(a + d_O) / gamma(a)
    EM_coeff = (
        bernoulli(2 * order)[2::2] / factorial(np.arange(1, order + 1) * 2)
    ).reshape((order,) + (1,) * a.ndim)
    return np.sum((derivatives_M - derivatives_kp1) * EM_coeff, axis=0)


def log_zeta_larger_than_1(a, log_M, order, k):

    # Partial Sum
    partial_sum = partial_sum_zeta(a, 1, k)

    # Boundary coerrections
    correction = ((k + 1) ** (-a) + np.exp2(-a * log_M)) / 2

    # Integral term
    integral = (np.exp2((1 - a) * log_M) - (k + 1) ** (1 - a)) / (1 - a)

    # Euler Maclaurin correction
    correction += euler_maclaurin_correction(a, k, log_M, order)

    return np.log(partial_sum + integral + correction)


def log_zeta_less_than_1(a, log_M, order, k):
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


def log_zeta(a: np.ndarray[float], log_M: float, order=10, k=50):

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


def log_guessing_entropy(mutual_information, key_size, base=2, order=10, k=50):
    r"""Output a lower bound on the logarithm in base 'base' of the log_guessing_entropy rank of the key of size key_size
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary.

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
    """
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
    return np.maximum(0, LG / np.log(base))


def median(
    mutual_information: np.ndarray[float], key_size: int, base: float = 2
) -> np.ndarray[float]:
    r"""Output a lower bound on the logarithm in base 'base' of the median rank of the key of size key_size
    when a leakage upper bounded by the mutual information 'mutual_information' is disclosed to the adversary.

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
    """

    # We use log1p and exp1m to have a stable expression
    # We deal with eventual overflows in the exponential for large mutual information next
    median = (key_size - 1) / np.log2(base) + np.log1p(
        -np.sqrt(-np.expm1(-2 * mutual_information * np.log(base)))
    ) / np.log(base)

    # To avoid numerical overflows we use an assymptotic expansion for large values of the mutual information
    large_mi = mutual_information > 12
    expansion_for_large_mi = np.maximum(
        0, (key_size - 2) / np.log2(base) - 2 * mutual_information
    )

    return np.where(large_mi, expansion_for_large_mi, median)
