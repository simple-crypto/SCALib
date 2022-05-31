import numpy as np
from scalib import _scalib_ext
from scalib.config.threading import _get_threadpool


class Ttest:
    r"""The `Ttest` object enables to perform univariate :math:`t`-test.
    Concretely, it first computes the  :math:`\mu[j]`'s and :math:`v[j]`'s for
    the `d`-th statistical moment to test at all the indexes (`j`) in the
    traces. These are derived based on all the provided traces :math:`l[:,j]`.
    These traces can be provided through multiple calls to `fit_u`.
    Second, the statistic `t[j]` is derived from the above equation.

    For `d=1`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} l[i,j]

    Similarly if `d=2`:

    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} (l[i,j] - \bar{l}[:,j])^2

    And finally if `d>2`:


    .. math::
        \mu[j] = \frac{1}{n} \sum_{i=0}^{n-1} \left(\frac{l[i,j] - \bar{l}[:,j])}{\sigma_{l[:,j]}}\right)^d

    Especially, :math:`\bar{l}` denotes the estimated mean of `l` and
    :math:`\sigma_l` its standard deviation.

    Parameters
    ----------
    ns : int
        Number of samples in a single trace.
    d : int
        Maximal statistical order of the :math:`t`-test.

    Examples
    --------
    >>> from scalib.metrics import Ttest
    >>> import numpy as np
    >>> traces = np.random.randint(0,256,(100,200),dtype=np.int16)
    >>> X = np.random.randint(0,2,100,dtype=np.uint16)
    >>> ttest = Ttest(200,d=3)
    >>> ttest.fit_u(traces,X)
    >>> t = ttest.get_ttest()

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

        self._ttest.update(l, x, _get_threadpool())

    def get_ttest(self):
        r"""Return the current Ttest estimation with an array of shape `(d,ns)`."""
        return self._ttest.get_ttest(_get_threadpool())


class MTtest:
    r"""The `MTtest` object enables to perform multivariate :math:`t`-test.
    Concretely, it first computes the  :math:`\mu[j]`'s and :math:`v[j]`'s to
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

        self._mttest.update(l, x, _get_threadpool())

    def get_ttest(self):
        r"""Return the current MTtest estimation with an array of shape
        `(n_pois,)`. Each element in that array corresponds to a test defined by
        `pois`.

        """
        return self._mttest.get_ttest(_get_threadpool())
