import warnings

import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.tools
import scalib.utils


class LDAClassifier:
    r"""Models the leakage :math:`\mathbf{l}` with :math:`n_s` dimensions using
    the linear discriminant analysis classifier (LDA) with integrated
    dimensionality reduction.

    .. deprecated:: 0.6.1
            Use ``LdaAcc`` instead.

    Based on the training data, linear discriminant analysis build a linear
    dimentionality reduction to :math:`p` dimensions that maximizes class
    separation.
    Then, a multivariate gaussian template is fitted for each class (using the
    same covariance matrix for all the classes) in the reduced dimensionality
    space to predict leakage likelihood :footcite:p:`LDA`.

    Let :math:`\mathbf{W}` be the dimensionality reduction matrix of size
    (:math:`p`, :math:`n_s`). The likelihood is

    .. math::
            \mathsf{\hat{f}}(\mathbf{l} | x) =
                \frac{1}{\sqrt{(2\pi)^{p} \cdot |\mathbf{\Sigma} |}} \cdot
                \exp^{\frac{1}{2}
                    (\mathbf{W} \cdot \mathbf{l} - \mathbf{\mu}_x)
                    \mathbf{\Sigma}
                    ( \mathbf{W} \cdot \mathbf{l}-\mathbf{\mu}_x)'}

    where :math:`\mathbf{\mu}_x` is the mean of the leakage for class :math:`x` in
    the projected space (:math:`\mu_x = \mathbb{E}(\mathbf{W}\mathbf{l}_x)`, where
    :math:`\mathbf{l}_x` denotes the leakage traces of class :math:`x`) and
    :math:`\mathbf{\Sigma}` its covariance (:math:`\mathbf{\Sigma} =
    \mathbb{Cov}(\mathbf{W}\mathbf{l}_x - \mathbf{\mu}_x)`).

    :class:`LDAClassifier` provides the probability of each class with :meth:`predict_proba`
    thanks to Bayes' law such that

    .. math::
        \hat{\mathsf{pr}}(x|\mathbf{l}) = \frac{\hat{\mathsf{f}}(\mathbf{l}|x)}
                    {\sum_{x^*=0}^{n_c-1} \hat{\mathsf{f}}(\mathbf{l}|x^*)}.

    Example
    -------
    >>> from scalib.modeling import LDAClassifier
    >>> import numpy as np
    >>> # 5000 traces of length 10, with value between 0 and 255
    >>> traces = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> # classes between 0 and 15
    >>> x = np.random.randint(0,16,5000,dtype=np.uint16)
    >>> lda = LDAClassifier(16,3)
    >>> lda.fit_u(traces, x)
    >>> lda.solve()
    >>> # predict classes for new traces
    >>> nt = np.random.randint(0,256,(20,10),dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(nt)

    Notes
    -----
    This should have similar behavior as scikit-learn's `LDA
    <https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda>`_, but it
    has better performance and numerical properties (at the cost of
    flexibility).

    References
    ----------

    .. footbibliography::

    Parameters
    ----------
    nc :
        Number of possible classes (e.g., 256 for 8-bit target). ``nc`` must
        be smaller than :math:`2^{16}`.
    p :
        Number of dimensions in the linear subspace.
    """

    def __init__(self, nc: int, p: int):
        self.solved = False
        self.done = False
        self.p = p
        self._nc = nc
        self._ns = None
        self._init = False
        if p >= nc:
            raise ValueError("p must be at most nc")
        warnings.warn(
            "LdaClassifier is deprecated. Use LdaAcc instead.", DeprecationWarning
        )

    def fit_u(
        self,
        traces: npt.NDArray[np.int16],
        x: npt.NDArray[np.uint16],
        gemm_mode: int | None = None,
    ):
        r"""Update statistical model estimates with fresh data.

        Parameters
        ----------
        traces :
            Array that contains the traces. The array must
            be of dimension ``(n,ns)`` and its type must be `int16`.
        x :
            Labels for each trace. Must be of shape ``(n)`` and
            must be `uint16`.
        gemm_mode:
            Depreciated, kept for API compatibility.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x[:, np.newaxis], nv=1, multi=True)
        if self.done:
            raise ValueError("Cannot fit_u after calling .solve(..., done=True).")
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self.acc = _scalib_ext.MultiLdaAcc(
                self._ns, self._nc, np.arange(self._ns, dtype=np.uint32)[np.newaxis, :]
            )
        # TODO maybe there is something smarter to do here w.r.t. number of
        # threads + investigate exact BLIS behavior.
        with scalib.utils.interruptible():
            self.acc.fit(traces, x, get_config())
        self.solved = False

    def solve(self, done: bool = False):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\mathbf{W}`, the means :math:`\mathbf{\mu}_x` and the covariance
        :math:`\mathbf{\Sigma}`.

        Parameters
        ----------
        done :
            True if the object will not be futher updated (clears some internal
            state, saving memory).

        Notes
        -----
        Once this has been called, predictions can be performed.
        """
        if not self._init:
            raise ValueError("Cannot .solve since .fit_u was never called.")
        if self.solved:
            raise ValueError(
                "Already called .solve() on this object, should not be called twice."
            )
        with scalib.utils.interruptible():
            self.mlda = self.acc.multi_lda(self.p, get_config())
        self.solved = True
        self.done = done

        if done:
            del self.acc

    def predict_proba(self, traces: npt.NDArray[np.int16]) -> npt.NDArray[np.float64]:
        r"""Computes the probability for each of the classes for the traces.

        Parameters
        ----------
        traces :
            Array that contains the traces. The array must be of dimension ``(n,ns)``.

        Returns
        -------
        array_like, f64
            Probabilities. Shape ``(n, nc)``.
        """
        if not self.solved:
            raise ValueError(
                "Call LDA.solve() before LDA.predict_proba() to compute the model."
            )
        with scalib.utils.interruptible():
            prs = self.mlda.predict_proba(traces, False, get_config())[0]
        return prs

    def _raw_scores(self, traces: npt.NDArray[np.int16]) -> npt.NDArray[np.float64]:
        if not self.solved:
            raise ValueError(
                "Call LDA.solve() before LDA.predict_proba() to compute the model."
            )
        with scalib.utils.interruptible():
            prs = self.mlda.predict_proba(traces, True, get_config())[0]
        return prs

    def get_sw(self):
        r"""Return :math:`S_{W}` matrix (within-class scatter)."""
        return self.acc.get_sw()[0]

    def get_sb(self):
        r"""Return :math:`S_{B}` matrix (between-class scatter)."""
        return self.acc.get_sb()[0]

    def get_mus(self):
        r"""Return means matrix (classes means). Shape: ``(nc, ns)``."""
        return self.acc.get_mus()[0]


class MultiLDA:
    """Perform LDA on `nv` distinct variables for the same leakage traces.

    .. deprecated:: 0.6.1
            Use ``LdaAcc`` instead.

    While functionally similar to a simple for loop, this enables solving the
    LDA problems in parallel in a simple fashion. This also enable easy
    handling of Points Of Interest (POIs) in long traces.

    Parameters
    ----------
    ncs: array_like, int
        Number of classes for each variable. Shape ``(nv,)``.
    ps: array_like, int
        Number of dimensions to keep after dimensionality reduction for each variable.
        Shape ``(nv,)``.
    pois: list of array_like, int
        Indices of the POIs in the traces for each variable. That is, for
        variable ``i``, and training trace ``t``, ``t[pois[i]]`` is the input
        datapoints for the LDA.
    gemm_mode:
        See :func:`LDACLassifier.fit_u`.

    Examples
    --------
    >>> from scalib.modeling import MultiLDA
    >>> import numpy as np
    >>> # 5000 traces with 50 points each
    >>> traces = np.random.randint(0, 256, (5000,50),dtype=np.int16)
    >>> # 5 variables (8-bit), and 5000 traces
    >>> x = np.random.randint(0, 256, (5000, 5),dtype=np.uint16)
    >>> # 10 POIs for each of the 5 variables
    >>> pois = [list(range(7*i, 7*i+10)) for i in range(5)]
    >>> # Keep 3 dimensions after dimensionality reduction
    >>> lda = MultiLDA(5*[256], 5*[3], pois)
    >>> lda.fit_u(traces, x)
    >>> lda.solve()
    >>> # Predict the class for 20 traces.
    >>> nt = np.random.randint(0, 256, (20, 50), dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(nt)
    """

    def __init__(self, ncs, ps, pois, gemm_mode: int | None = None):
        self.pois = pois
        self.ldas = [LDAClassifier(nc, p) for nc, p in zip(ncs, ps)]
        warnings.warn("MultiLDA is deprecated. Use LdaAcc instead.", DeprecationWarning)

    def fit_u(self, traces, x):
        """Update the LDA estimates with new training data.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must
            be of dimension ``(n,ns)`` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape ``(n, nv)`` and
            must be `uint16`.
        """
        # Try to avoid the over-subscription of CPUs.
        num_threads = get_config().threadpool.n_threads
        with scalib.utils.interruptible():
            with scalib.tools.ContextExecutor(max_workers=num_threads) as executor:
                list(
                    executor.map(
                        lambda i: self.ldas[i].fit_u(
                            np.ascontiguousarray(traces[:, self.pois[i]]),
                            x[:, i],
                        ),
                        range(len(self.ldas)),
                    )
                )

    def solve(self, done: bool = False):
        """See `LDAClassifier.solve`."""
        # Put as much work as needed to fill rayon threadpool
        with scalib.utils.interruptible():
            with scalib.tools.ContextExecutor(
                max_workers=get_config().threadpool.n_threads
            ) as executor:
                list(executor.map(lambda lda: lda.solve(done), self.ldas))

    def predict_proba(self, traces):
        """Predict probabilities for all variables.

        See `LDAClassifier.predict_proba`.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must
            be of dimension ``(n,ns)``.

        Returns
        -------
        list of array_like, f64
            Probabilities. ``nv`` arrays of shape ``(n, nc)``.
        """
        # Put as much work as needed to fill rayon threadpool
        with scalib.utils.interruptible():
            with scalib.tools.ContextExecutor(
                max_workers=get_config().threadpool.n_threads
            ) as executor:
                return list(
                    executor.map(
                        lambda i: self.ldas[i].predict_proba(traces[:, self.pois[i]]),
                        range(len(self.ldas)),
                    )
                )

    def _raw_scores(self, traces):
        return [
            self.ldas[i]._raw_scores(traces[:, self.pois[i]])
            for i in range(len(self.ldas))
        ]

    def get_sw(self):
        return [lda.get_sw() for lda in self.ldas]

    def get_sb(self):
        return [lda.get_sb() for lda in self.ldas]

    def get_mus(self):
        return [lda.get_mus() for lda in self.ldas]


class LdaAcc:
    r"""Models the leakage :math:`\mathbf{l}` with :math:`n_s` dimensions using
    the linear discriminant analysis classifier (LDA) with integrated
    dimensionality reduction.

    Based on the training data, linear discriminant analysis build a linear
    dimentionality reduction to :math:`p` dimensions that maximizes class
    separation, for each target variable.
    Then, for each variable, a multivariate gaussian template is fitted for
    each class (using the same covariance matrix for all the classes) in the
    reduced dimensionality space to predict leakage likelihood
    :footcite:p:`LDA`.

    Let :math:`\mathbf{W}` be the dimensionality reduction matrix of size
    (:math:`p`, :math:`n_s`). The likelihood is

    .. math::
            \mathsf{\hat{f}}(\mathbf{l} | x) =
                \frac{1}{\sqrt{(2\pi)^{p} \cdot |\mathbf{\Sigma} |}} \cdot
                \exp^{\frac{1}{2}
                    (\mathbf{W} \cdot \mathbf{l} - \mathbf{\mu}_x)
                    \mathbf{\Sigma}
                    ( \mathbf{W} \cdot \mathbf{l}-\mathbf{\mu}_x)'}

    where :math:`\mathbf{\mu}_x` is the mean of the leakage for class :math:`x` in
    the projected space (:math:`\mu_x = \mathbb{E}(\mathbf{W}\mathbf{l}_x)`, where
    :math:`\mathbf{l}_x` denotes the leakage traces of class :math:`x`) and
    :math:`\mathbf{\Sigma}` its covariance (:math:`\mathbf{\Sigma} =
    \mathbb{Cov}(\mathbf{W}\mathbf{l}_x - \mathbf{\mu}_x)`).

    This class is aimed to be used together with :class:`Lda`. In particular, :class:`LdaAcc`
    only perform the fitting step, from which a fitted :class:`Lda` instance can be created
    for an arbitrary :math:`p`. The solved :class:`Lda` instance provides the probability of
    each class with :meth:`predict_proba` thanks to Bayes' law such that

    .. math::
        \hat{\mathsf{pr}}(x|\mathbf{l}) = \frac{\hat{\mathsf{f}}(\mathbf{l}|x)}
                    {\sum_{x^*=0}^{n_c-1} \hat{\mathsf{f}}(\mathbf{l}|x^*)}.

    Example
    -------
    >>> from scalib.modeling import Lda, LdaAcc
    >>> import numpy as np
    >>> # 1000 traces of length 10, with value between 0 and 4091
    >>> traces = np.random.randint(0, 4092, (1000, 10), dtype=np.int16)
    >>> # classes between 0 and 15 (2 variables)
    >>> x = np.random.randint(0, 16, (1000, 2), dtype=np.uint16)
    >>> lda_acc = LdaAcc(nc=16, pois=[list(range(5)), list(range(4, 10))])
    >>> lda_acc.fit_u(traces, x)
    >>> lda = Lda(lda_acc, p=3) # Projection to 3 dimensions.
    >>> # predict classes for new traces
    >>> new_traces = np.random.randint(0, 256, (20,10), dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(new_traces)

    Notes
    -----
    This should have similar behavior as scikit-learn's `LDA
    <https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda>`_, but it
    has better performance and numerical properties (at the cost of
    flexibility).

    References
    ----------

    .. footbibliography::

    Parameters
    ----------
    nc: int
        Number of possible classes (e.g., 256 for 8-bit target). ``nc`` must
        be smaller than :math:`2^{16}`.
    pois: list of array_like, int
        Indices of the POIs in the traces for each variable. That is, for
        variable ``i``, and training trace ``t``, ``t[pois[i]]`` is the input
        datapoints for the LDA.
    """

    def __init__(self, nc: int, pois):
        self._nc = nc
        self._pois = pois
        self._nv = len(self._pois)
        self._init = False

    def fit_u(self, traces, x):
        """Update the LDA estimates with new training data.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must
            be of dimension ``(n,ns)`` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape ``(n, nv)`` and
            must be `uint16`.
        """
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self._inner = _scalib_ext.MultiLdaAcc(self._ns, self._nc, self._pois)
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, self._nv)
        with scalib.utils.interruptible():
            self._inner.fit(traces, x, get_config())

    def get_sw(self):
        r"""Return :math:`S_{W}` matrix (within-class scatter)."""
        return self._inner.get_sw()

    def get_sb(self):
        r"""Return :math:`S_{B}` matrix (between-class scatter)."""
        return self._inner.get_sb()

    def get_mus(self):
        r"""Return means matrix (classes means) as a list of length ``nv``, where the ``i``-th element is the matrix of shape: ``(nc, len(pois[i]))`` associated to the ``i``-th variable."""
        return self._inner.get_mus()


class Lda:
    """See :class:`LdaAcc`."""

    def __init__(self, acc: LdaAcc, p: int):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\mathbf{W}`, the means :math:`\mathbf{\mu}_x` and the covariance
        :math:`\mathbf{\Sigma}`.

        Parameters
        ----------
        p :
            Number of dimensions to keep after dimensionality reduction for each variable.
        acc :
            Fitted :class:`LdaAcc` instance.
        """
        if not acc._init:
            raise ValueError("Empty accumulator: .fit_u was never called.")
        with scalib.utils.interruptible():
            self._inner = acc._inner.multi_lda(p, get_config())
        self._nv = acc._nv
        self._ns = acc._ns

    def predict_proba(self, traces):
        r"""Computes the probability for each of the classes for the traces,
            for all variables.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must be of dimension ``(n,ns)``.

        Returns
        -------
        list of array_like, f64
            Probability distributions. Shape ``(nv, n, nc)``.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        with scalib.utils.interruptible():
            return self._inner.predict_proba(traces, False, get_config())

    def _raw_scores(self, traces):
        r"""Raw scores, i.e., predict_proba w/o softmax (for tests)."""
        traces = scalib.utils.clean_traces(traces, self._ns)
        with scalib.utils.interruptible():
            return self._inner.predict_proba(traces, True, get_config())

    def predict_log2_proba_class(self, traces, x):
        r"""Computes the log2 probability for the given class for the traces,
            for all variables.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must be of dimension ``(n,ns)``.
        x : array_like, uint16
            Labels for each trace. Must be of shape ``(n, nv)`` and
            must be `uint16`.

        Returns
        -------
        list of array_like, f64
            Log2 probabilities. Shape ``(nv, n)``.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, self._nv)

        with scalib.utils.interruptible():
            return self._inner.predict_log2_proba_class(traces, x, get_config())

    def project(self, traces: npt.NDArray[np.int16]) -> npt.NDArray[np.float64]:
        r"""Project the traces in the sub-linear space.

        Parameters
        ----------
        traces:
            Array that contains the traces. The array must be of dimension ``(n,ns)``.

        Returns
        -------
        array_like, f64
            Projected traces. List of ``nv`` array of shape ``(n, self.p)``
        """
        return self._inner.project(traces, get_config())

    def select_vars(self, vars: list[int]) -> "Lda":
        r"""Make a new :class:`Lda` with only a subset of the variables (in the
        order given by the list).

        Parameters
        ----------
        traces :
            List of selected variables.

        Returns
        -------
        Lda
            A new Lda.
        """
        cls = type(self)
        new = cls.__new__(cls)
        new._inner = self._inner.select_vars(vars)
        new._nv = len(vars)
        new._ns = self._ns
        return new
