r"""
The Student's :math:`t`-test can be used to highlight a difference in the means
of two distributions. To do so, a `t` statistic is derived following the
expression:

.. math::
    t = \frac{\mu_0 - \mu_1}{\sqrt{\frac{v_0}{n_0} + \frac{v_1}{n_1}}}

where :math:`\mu_0` (resp. :math:`\mu_1`) is the estimated moment of the first
(resp.second) population and :math:`\frac{v_0}{n_0}` the variance of its
estimate from :math:`n_0` samples. In the context of side-channel analysis, many of
these statistical tests are performed independently for each point of the traces. 
See [1]_ for additional details.

In this module, the definition of :math:`\mu` and :math:`v` are adapted to perform
higher-order univariate and multivariate :math:`t`-test to compare higher-order moments of
two distributions:

- *Higher-order t-tests* pre-process the populations by elevating the traces at
  the :math:`d`-th power, after subtracting the mean (for :math:`d>1`) and
  normalizing the variance (for :math:`d>2`).
- *Multivariate t-test* are a variant of higher-order t-tests where a product
  of :math:`d` samples of the trace is use instead of the :math:`d`-th power of
  a single sample. Mean and variance normalization are applied similarly to
  higher-order t-tests.

.. currentmodule:: scalib.metrics

.. autosummary::
    :toctree:
    :nosignatures:
    :recursive:

    Ttest
    MTtest

**Warning**: Ttest should not be used alone as a standalone evaluation tool
because of its qualitative nature. See [2]_ and [3]_ for cautionary notes.


Implementations Details
^^^^^^^^^^^^^^^^^^^^^^^

In order to enable both efficient one-core and parallelized performance of the
:math:`t`-test implementation, SCALib uses the one-pass formula for estimation
arbitrary order statistical moment from [4]_ and its application to
side-channel context in [1]_.

Concretely, the implementations first compute an estimation of the required
statistical moments using a two-passes algorithms (first pass to compute the
mean and the variances, and a second pass to compute the centered products).
This new estimation is then used to update the current estimation using the
merging rule from [4]_. To enable multi-threading, SCALib internally divides 
the fresh traces into smaller independent chunks and then merges the output of
each threads using [4]_. 

As a conclusion, the performance of the SCALib improves if the two-passes
algorithm can be used on large chunks. Hence, it is recommended to feed a large
enough amount of data for every call to `fit_u()`. 

References
^^^^^^^^^^

.. [1] "Leakage assessment methodology", Tobias Schneider, Amir Moradi, CHES
   2015
.. [2] "How (not) to Use Welch’s T-test in Side-Channel Security
   Evaluations", François-Xavier Standaert, CARDIS 2018
.. [3] "A Critical Analysis of ISO 17825 ('Testing Methods for the
   Mitigation of Non-invasive Attack Classes Against Cryptographic
   Modules')", Carolyn Whitnall, Elisabeth Oswald, ASIACRYPT 2019
.. [4] "Formulas for Robust, One-Pass Parallel Computation of Covariances and
    Arbitrary-Order Statistical Moments", Philippe Pébay, 2008

"""
import numpy as np

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.utils


class Ttest:
    r"""Univariate (higher order) :math:`t`-test.

    Concretely, :math:`\mu[j]`'s and :math:`v[j]`'s are computed for
    the `d`-th statistical moment to test at all the indexes (`j`) in the
    traces. These are derived based on all the provided traces :math:`l[:,j]`, that
    can be provided through multiple calls to `fit_u`.

    The statistic `t[j]` is then derived as:

    - for `d=1`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} l[i,j]

    - for `d=2`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} (l[i,j] - \bar{l}[:,j])^2

    - for `d>2`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} \left(\frac{l[i,j] - \bar{l}[:,j])}{\sigma_{l[:,j]}}\right)^d

    where, :math:`\bar{l}` denotes the estimated mean of `l` and
    :math:`\sigma_l` its standard deviation.

    A :math:`d`-th order t-test automatically includes the computation of all the lower order t-tests.

    Example
    --------
    >>> from scalib.metrics import Ttest
    >>> import numpy as np
    >>> traces = np.random.randint(0,256,(100,200),dtype=np.int16)
    >>> X = np.random.randint(0,2,100,dtype=np.uint16)
    >>> ttest = Ttest(200,d=3)
    >>> ttest.fit_u(traces,X)
    >>> t = ttest.get_ttest()

    Parameters
    ----------
    ns : int
        Number of samples in a single trace.
    d : int
        Maximal statistical order of the :math:`t`-test.
    """

    def __init__(self, ns, d):
        self._ns = ns
        self._d = d

        self._ttest = _scalib_ext.Ttest(ns, d)

    def fit_u(self, l, x):
        r"""Updates the Ttest estimation with samples of `l` for the sets `x`.

        This method may be called multiple times.

        Parameters
        ----------
        l : array_like, np.int16
            Array that contains the signal. The array must
            be of dimension `(n, ns)` and its type must be `np.int16`.
        x : array_like, np.uint16
            Set in which each trace belongs. Must be of shape `(n,)`, must be
            `np.uint16` and must contain only `0` and `1`.
        """
        nl, nsl = l.shape
        nx = x.shape[0]
        if not (nx == nl):
            raise ValueError(f"Expected x with shape ({nl},)")
        if not (nsl == self._ns):
            raise Exception(f"Expected second dim of l to have size {self._ns}.")

        with scalib.utils.interruptible():
            self._ttest.update(l, x, get_config())

    def get_ttest(self):
        r"""Return the current Ttest estimation with an array of shape `(d,ns)`."""
        with scalib.utils.interruptible():
            return self._ttest.get_ttest(get_config())


class MTtest:
    r"""Multivariate :math:`t`-test.

    Concretely, :math:`\mu[j]`'s and :math:`v[j]`'s are computed to
    use in the :math:`t`-test definition. Especially, :math:`\mu[j]`'s is
    derived such as:

    For `d=2`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} \prod_{d'=0}^{1}l[i,pois[d',j]] - \bar{l}[:,pois[d',j]]

    For `d>2`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} \prod_{d'=0}^{d-1}
        \frac{l[i,pois[d',j]] -
        \bar{l}[:,pois[d',j]]}{\sigma_{l[:,pois[d',j]]}}

    Especially, :math:`\bar{l}` denotes the estimated mean of `l` and
    :math:`\sigma_l` its standard deviation. `pois` defines the points in the
    traces for which the (normalized) product will be tested.

    Parameters
    ----------
    d : int
        Maximal statistical order of the :math:`t`-test.
    pois : array_like, uint32
        Array of share `(d,n_pois)`. Each column in `pois` will result in a
        :math:`t`-test of the product of the traces at these $d$ indexes.

    Examples
    --------
    >>> from scalib.metrics import MTtest
    >>> import numpy as np
    >>> traces = np.random.randint(0,256,(100,200),dtype=np.int16)
    >>> pois = np.random.randint(0,200,(2,5000),dtype=np.uint32)
    >>> X = np.random.randint(0,2,100,dtype=np.uint16)
    >>> mttest = MTtest(d=2,pois=pois)
    >>> mttest.fit_u(traces,X)
    >>> t = mttest.get_ttest()

    """

    def __init__(self, d, pois):
        self._pois = pois
        self._d = d
        self._ns = len(pois[0, :])

        self._mttest = _scalib_ext.MTtest(d, self._pois)

    def fit_u(self, l, x):
        r"""Updates the MTtest estimation with samples of `l` for the sets `x`.
        This method may be called multiple times.

        Parameters
        ----------
        l : array_like, np.int16
            Array that contains the signal. The array must
            be of dimension `(n, ns)` and its type must be `np.int16`.
        x : array_like, np.uint16
            Set in which each trace belongs. Must be of shape `(n,)`, must be
            `np.uint16` and must contain only `0` and `1`.
        """
        nl, _ = l.shape
        nx = x.shape[0]
        if not (nx == nl):
            raise ValueError(f"Expected x with shape ({nl},)")

        with scalib.utils.interruptible():
            self._mttest.update(l, x, get_config())

    def get_ttest(self):
        r"""Return the current MTtest estimation with an array of shape
        `(n_pois,)`. Each element in that array corresponds to a test defined by
        `pois`.

        """
        with scalib.utils.interruptible():
            return self._mttest.get_ttest(get_config())
