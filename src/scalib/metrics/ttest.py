import numpy as np
from scalib import _scalib_ext


class Ttest:
    r"""Computes the univariate :math:`t`-test at arbitrary order :math:`d`
    between two sets :math:`i` of traces. Informally, it allows to highlight a
    difference in statistical moments of order :math:`d` between the two sets.
    The metric is defined with:

    .. math::
        t = \frac{x_0 - x_1}{\sqrt{(v_0/n_0)+(v_1/n_1)}}

    where the both :math:`x_i` and :math:`v_i` are defined independently for
    each of the two sets and :math:`n_i` the number of available samples in the
    set :math:`i`.

    The expressions of :math:`x_i` and :math:`v_i` depend on the order `d` and relies on
    estimation of central and standardized moments. See [1]_ for full details.

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

    Notes
    -----
    **Warning**: Ttest should not be used alone as a standalone evaluation tool
    because of its qualitative nature. See [2]_ and [3]_ for cautionary notes.

    .. [1] "Leakage assessment methodology", Tobias Schneider, Amir Moradi, CHES
       2015
    .. [2] "How (not) to Use Welch’s T-test in Side-Channel Security
       Evaluations", François-Xavier Standaert, CARDIS 2018
    .. [3] "A Critical Analysis of ISO 17825 ('Testing Methods for the
       Mitigation of Non-invasive Attack Classes Against Cryptographic
       Modules')", Carolyn Whitnall, Elisabeth Oswald, ASIACRYPT 2019

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

        self._ttest.update(l, x)

    def get_ttest(self):
        r"""Return the current Ttest estimation with an array of shape `(d,ns)`."""
        return self._ttest.get_ttest()


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
        nl, nsl = l.shape
        nx = x.shape[0]
        if not (nx == nl):
            raise ValueError(f"Expected x with shape ({nl},)")
        # if not (nsl == self._ns):
        #    raise Exception(f"Expected second dim of l to have size {self._ns}.")

        self._mttest.update(l, x)

    def get_ttest(self):
        r"""Return the current MTtest estimation with an array of shape
        `(n_pois,)`. Each element in that array corresponds to a test defined by
        `pois`.

        """
        return self._mttest.get_ttest()
