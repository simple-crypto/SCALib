r"""
The Welch's :math:`t`-test can be used to highlight a difference in the means
of two distributions. To do so, a `t` statistic is derived following the
expression:

.. math::
    t = \frac{\mu_0 - \mu_1}{\sqrt{\frac{v_0}{n_0} + \frac{v_1}{n_1}}}

where :math:`\mu_0` (resp. :math:`\mu_1`) is the estimated moment of the first
(resp.second) population and :math:`\frac{v_0}{n_0}` the variance of its
estimate from :math:`n_0` samples. In the context of side-channel analysis, many of
these statistical tests are performed independently for each point of the traces. 
See :footcite:p:`TVLA` for additional details.

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

Warning
^^^^^^^

Ttest should not be used alone as a standalone evaluation tool because of its
qualitative nature. See :footcite:p:`how_not_ttest,critical_analysis_iso17825`
for cautionary notes.


Implementations Details
^^^^^^^^^^^^^^^^^^^^^^^

In order to enable both efficient one-core and parallelized performance of the
:math:`t`-test implementation, SCALib uses the one-pass formula for estimation
arbitrary order statistical moment from :footcite:p:`onepass_moments` and its
application to side-channel context in :footcite:p:`TVLA`.

Concretely, the implementations first compute an estimation of the required
statistical moments using a two-passes algorithms (first pass to compute the
mean and the variances, and a second pass to compute the centered products).
This new estimation is then used to update the current estimation using the
merging rule from :footcite:p:`onepass_moments`. To enable multi-threading,
SCALib internally divides the fresh traces into smaller independent chunks and
then merges the output of each threads using :footcite:p:`onepass_moments`. 

As a conclusion, the performance of the SCALib improves if the two-passes
algorithm can be used on large chunks. Hence, it is recommended to feed a large
enough amount of data for every call to `fit_u()`. 

References
^^^^^^^^^^

.. footbibliography::
"""

import numpy as np
import numpy.typing as npt

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
    >>> x = np.random.randint(0,2,100,dtype=np.uint16)
    >>> ttest = Ttest(d=3)
    >>> ttest.fit_u(traces,x)
    >>> t = ttest.get_ttest()

    Parameters
    ----------
    d : int
        Maximal statistical order of the :math:`t`-test.
    """

    def __init__(self, d: int):
        self._d = d
        self._ns = None
        self._init = False

    def fit_u(self, traces: npt.NDArray[np.int16], x: npt.NDArray[np.uint16]):
        r"""Updates the Ttest estimation with new data.

        This method may be called multiple times.

        Parameters
        ----------
        traces :
            Array that contains the traces. The array must be of dimension
            `(n, ns)`.
        x :
            Set in which each trace belongs. Must be of shape `(n,)` and must
            contain only `0` and `1`.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, multi=False)
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self._ttest = _scalib_ext.Ttest(self._ns, self._d)
        if traces.shape[0] != x.shape[0]:
            raise ValueError(
                f"Number of traces {traces.shape[0]} does not match size of classes array {x.shape[0]}."
            )
        with scalib.utils.interruptible():
            self._ttest.update(traces, x, get_config())

    def get_ttest(self) -> npt.NDArray[np.float64]:
        r"""Return the current Ttest estimation with an array of shape `(d,ns)`."""
        if not self._init:
            raise ValueError("Need to call .fit_u at least once.")
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
        Number of variables of the :math:`t`-test.
    pois : array_like, uint32
        Array of shape ``(d,n_pois)``. Each column in `pois` will result in a
        :math:`t`-test of the product of the traces at these :math:`d` indexes
        (each index is between 0 and ns-1).
        If an index is repeated in a column, the corresponding test uses a
        higher-order moment for the corresponding point in the trace.

    Examples
    --------
    >>> from scalib.metrics import MTtest
    >>> import numpy as np
    >>> traces = np.random.randint(0,256,(100,200),dtype=np.int16)
    >>> # Take as POIs each point in the trace combined with any of the 10 following samples.
    >>> pois = np.array([[x, x+d] for x in range(200) for d in range(10) if x + d < 200], dtype=np.uint32).T
    >>> x = np.random.randint(0,2,100,dtype=np.uint16)
    >>> mttest = MTtest(d=2,pois=pois)
    >>> mttest.fit_u(traces,x)
    >>> t = mttest.get_ttest()

    """

    def __init__(self, d, pois):
        self._pois = pois
        self._d = d
        self._ns = len(pois[0, :])

        self._mttest = _scalib_ext.MTtest(d, self._pois)

    def fit_u(self, traces: npt.NDArray[np.int16], x: npt.NDArray[np.uint16]):
        r"""Updates the MTtest estimation with new traces.
        This method may be called multiple times.

        Parameters
        ----------
        traces :
            Array that contains the signal. The array must
            be of dimension `(n, ns)`.
        x :
            Set in which each trace belongs. Must be of shape `(n,)`
            and must contain only `0` and `1`.
        """
        traces = scalib.utils.clean_traces(traces)
        x = scalib.utils.clean_labels(x, multi=False)
        if x.shape[0] != traces.shape[0]:
            raise ValueError(
                f"Number of traces {traces.shape[0]} does not match size of classes array {x.shape[0]}."
            )
        with scalib.utils.interruptible():
            self._mttest.update(traces, x, get_config())

    def get_ttest(self) -> npt.NDArray[np.float64]:
        r"""Return the current MTtest estimation with an array of shape
        `(n_pois,)`. Each element in that array corresponds to a test defined by
        `pois`.

        """
        with scalib.utils.interruptible():
            return self._mttest.get_ttest(get_config())
