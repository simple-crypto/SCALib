import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.utils


class HwLdaAcc:
    r"""Regression-based Linear Discriminant Analysis for HW.

    TODO: update

    Models the leakage using a regression-based linear discriminant analysis
    (RLDA) classifier :footcite:p:`RLDA`, which can efficiently handle long
    traces and large number of classes.

    In a nutshell, this model performs LDA with the class means modelled as
    linear regression based on the :math:`n_b` bits of the class value.
    Compared to the :class:`scalib.modeling.LDAClassifier`, this model will
    perform better when the number of classes is large and/or there are few
    profiling traces.

    Internally, it first estimates the coefficients of the linear regression,
    then computes a projection matrix that reduces the dimensionality of the
    gaussian template to :math:`p` dimensions and makes the covariance matrix
    the identity.

    It is then able to predict the leakage likelihood

    .. math::
        \hat{\mathsf{f}}[\mathbf{l}|X=x] =
        \alpha
        \exp\left(
        -\frac{1}{2} \lVert\mathbf{W}^T\mathbf{l} - \mathbf{A}\mathbf\beta(x)\rVert^2
        \right).

    Where :math:`\mathbf{W}` is the projection matrix, :math:`\mathbf{A}` the projected
    regression coefficients, and :math:`\mathbf{\beta(x)}` the coefficients of :math:`x`.
    The parameter :math:`\alpha = 1/\sqrt{(2\pi)^p\lvert\hat\Sigma_\mathbf{W}}\rvert` does
    not need to be calculated as it will get canceled out when applying Bayes' law.

    :class:`RLDAClassifier` provides the probability for each of the :math:`2^{n_b}`
    classes with :meth:`predict_proba`.

    Examples
    --------

    >>> from scalib.modeling import RLDAClassifier
    >>> import numpy as np
    >>> traces_model = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> labels_model = np.random.randint(0,256,(5000,1),dtype=np.uint64)
    >>> rlda = RLDAClassifier(8, 3)
    >>> rlda.fit_u(traces_model, labels_model)
    >>> rlda.solve()
    >>> traces_test = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> prs = rlda.predict_proba(traces_test, 0)

    References
    ----------

    .. footbibliography::
    """

    def __init__(self, nb: int):
        """
        Parameters
        ----------
        nb:
            Number of bits of the profiled variables.
        nv:
            Number of variables to profile
        p:
            Number of dimensions in the linear subspace.
        """
        self._ns = None
        self._nv = None
        self._nb = nb
        self._init = False

    def fit_u(self, traces: npt.NDArray[np.int16], x: npt.NDArray[np.uint64]):
        """Update statistical model estimates with additional data.

        This can be called multiple times, the state is accumulated.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. Shape ``(n,ns)``.
        x : array_like, uint64
            Labels for each trace. Shape ``(n,nv)``.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, self._nv, exp_type=np.uint64)
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self._nv = x.shape[1]
            self._inner = _scalib_ext.HwLdaAcc(self._nb, self._ns, self._nv)
        self._inner.fit(traces, x.T, get_config())


class HwLda:
    def __init__(self, acc: HwLdaAcc):
        if not acc._init:
            raise ValueError("Empty accumulator: .fit_u was never called.")
        with scalib.utils.interruptible():
            self._inner = acc._inner.lda(get_config())
        self._nv = acc._nv
        self._ns = acc._ns

    def predict_proba(
        self, traces: npt.NDArray[np.int16], var: int
    ) -> npt.NDArray[np.float64]:
        r"""Computes the probability for each of the classes for the requested variables.

        Parameters
        ----------
        traces:
            Array that contains the traces. Shape ``(n,ns)``.
        var:
            Id (position in the ``x`` array) of the variable for which the
            probabilities are computed.

        Returns
        -------
        array_like, f64
            Probabilities. Shape ``(n, nc)``.
        """
        return self._inner.predict_proba(traces, var, get_config())

    def predict_log2p1(
        self, traces: npt.NDArray[np.int16], labels: npt.NDArray[np.uint64]
    ) -> npt.NDArray[np.float64]:
        r"""Computes the log2 probability for each of the corresponding classes for the requested variable.

        Parameters
        ----------
        traces:
            Array that contains the traces. Shape ``(n,ns)``.
        labels:
            labels associated to each element taken by all variables, as an array of shape ``(n, nv)``.

        Returns
        -------
        array_like, f64
            Probabilities. Shape ``(nv,n)``.
        """
        return self._inner.predict_log2_proba_class(traces, labels.T, get_config())

    def predict_hw_probas(
        self, traces: npt.NDArray[np.int16]
    ) -> npt.NDArray[np.float64]:
        r"""Computes the probability for each of the corresponding HWs.

        Parameters
        ----------
        traces:
            Array that contains the traces. Shape ``(n,ns)``.

        Returns
        -------
        array_like, f64
            Probabilities. Shape ``(nv,n, 2**nb + 1)``.
        """
        return self._inner.predict_hw_probas(traces, get_config())

    def project(
        self, traces: npt.NDArray[np.int16], var: int
    ) -> npt.NDArray[np.float64]:
        return self._inner.project(traces, var, get_config())
