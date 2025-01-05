import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.tools
import scalib.utils

from typing import Optional


class LDAClassifier:
    r"""Models the leakage :math:`\mathbf{l}` with :math:`n_s` dimensions using
    the linear discriminant analysis classifier (LDA) with integrated
    dimensionality reduction.

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
        Number of possible classes (e.g., 256 for 8-bit target). `nc` must
        be smaller than `2**16`.
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

    def fit_u(
        self,
        traces: npt.NDArray[np.int16],
        x: npt.NDArray[np.uint16],
        gemm_mode: int = 1,
    ):
        r"""Update statistical model estimates with fresh data.

        Parameters
        ----------
        traces :
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x :
            Labels for each trace. Must be of shape `(n)` and
            must be `uint16`.
        gemm_mode:
            If 0: use matrixmultiply matrix multiplication.
            If n>0: use n threads with BLIS matrix multiplication.
            BLIS is only used on linux. Matrixmultiply is always used on other
            OSes.
            The BLIS threads (if > 1) do not belong to the SCALib threadpool.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, multi=False)
        if self.done:
            raise ValueError("Cannot fit_u after calling .solve(..., done=True).")
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self.acc = _scalib_ext.LdaAcc(self._nc, self._ns)
        # TODO maybe there is something smarter to do here w.r.t. number of
        # threads + investigate exact BLIS behavior.
        with scalib.utils.interruptible():
            self.acc.fit(traces, x, gemm_mode, get_config())
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
            self.lda = self.acc.lda(self.p, get_config())
        self.solved = True
        self.done = done

        if done:
            del self.acc

    def predict_proba(self, traces: npt.NDArray[np.int16]) -> npt.NDArray[np.float64]:
        r"""Computes the probability for each of the classes for the traces.

        Parameters
        ----------
        traces :
            Array that contains the traces. The array must be of dimension `(n,ns)`.

        Returns
        -------
        array_like, f64
            Probabilities. Shape `(n, nc)`.
        """
        if not self.solved:
            raise ValueError(
                "Call LDA.solve() before LDA.predict_proba() to compute the model."
            )
        with scalib.utils.interruptible():
            prs = self.lda.predict_proba(traces, get_config())
        return prs

    def __getstate__(self):
        dic = self.__dict__.copy()

        if "acc" in dic:
            dic["acc"] = dic["acc"].get_state()
        if "lda" in dic:
            dic["lda"] = dic["lda"].get_state()

        return dic

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        if "acc" in state:
            self.acc = _scalib_ext.LdaAcc.from_state(*state["acc"])
        if "lda" in state:
            self.lda = _scalib_ext.LDA.from_state(*state["lda"])

    def get_sw(self):
        r"""Return :math:`S_{W}` matrix (within-class scatter)."""
        return self.acc.get_sw()

    def get_sb(self):
        r"""Return :math:`S_{B}` matrix (between-class scatter)."""
        return self.acc.get_sb()

    def get_mus(self):
        r"""Return means matrix (classes means). Shape: ``(nc, ns)``."""
        return self.acc.get_mus()

    @classmethod
    def _from_inner_solved(cls, lda):
        self = cls.__new__(cls)
        self.lda = lda
        self.solved = True
        self.done = True
        return self


class MultiLDA:
    """Perform LDA on `nv` distinct variables for the same leakage traces.

    While functionally similar to a simple for loop, this enables solving the
    LDA problems in parallel in a simple fashion. This also enable easy
    handling of Points Of Interest (POIs) in long traces.

    Parameters
    ----------
    ncs: array_like, int
        Number of classes for each variable. Shape `(nv,)`.
    ps: array_like, int
        Number of dimensions to keep after dimensionality reduction for each variable.
        Shape `(nv,)`.
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

    def __init__(self, ncs, ps, pois, gemm_mode: int = 1):
        self.pois = [list(sorted(x)) for x in pois]
        self.gemm_mode = gemm_mode
        self.ldas = [LDAClassifier(nc, p) for nc, p in zip(ncs, ps)]

    @classmethod
    def from_ldas(cls, pois, ldas):
        self = cls.__new__(cls)
        self.pois = pois
        self.ldas = ldas
        return self

    def fit_u(self, traces, x):
        """Update the LDA estimates with new training data.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape `(n, nv)` and
            must be `uint16`.
        """
        # Try to avoid the over-subscription of CPUs.
        num_threads = get_config().threadpool.n_threads
        if self.gemm_mode != 0:
            num_threads = max(1, num_threads // self.gemm_mode)
        with scalib.utils.interruptible():
            with scalib.tools.ContextExecutor(max_workers=num_threads) as executor:
                list(
                    executor.map(
                        lambda i: self.ldas[i].fit_u(
                            np.ascontiguousarray(traces[:, self.pois[i]]),
                            x[:, i],
                            self.gemm_mode,
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
            be of dimension `(n,ns)`.

        Returns
        -------
        list of array_like, f64
            Probabilities. `nv` arrays of shape `(n, nc)`.
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

    def get_sw(self):
        return [lda.get_sw() for lda in self.ldas]

    def get_sb(self):
        return [lda.get_sb() for lda in self.ldas]

    def get_mus(self):
        return [lda.get_mus() for lda in self.ldas]


class Lda:
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

    :class:`Lda` provides the probability of each class with :meth:`predict_proba`
    thanks to Bayes' law such that

    .. math::
        \hat{\mathsf{pr}}(x|\mathbf{l}) = \frac{\hat{\mathsf{f}}(\mathbf{l}|x)}
                    {\sum_{x^*=0}^{n_c-1} \hat{\mathsf{f}}(\mathbf{l}|x^*)}.

    Example
    -------
    >>> from scalib.modeling import Lda
    >>> import numpy as np
    >>> # 1000 traces of length 10, with value between 0 and 4091
    >>> traces = np.random.randint(0, 4092, (1000, 10), dtype=np.int16)
    >>> # classes between 0 and 15 (2 variables)
    >>> x = np.random.randint(0, 16, (1000, 2), dtype=np.uint16)
    >>> lda = Lda(nc=16, pois=[list(range(5)), list(range(4, 10))])
    >>> lda.fit_u(traces, x)
    >>> lda.solve(p=3) # Projection to 3 dimensions.
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
        Number of possible classes (e.g., 256 for 8-bit target). `nc` must
        be smaller than `2**16`.
    pois: list of array_like, int
        Indices of the POIs in the traces for each variable. That is, for
        variable ``i``, and training trace ``t``, ``t[pois[i]]`` is the input
        datapoints for the LDA.
    """

    def __init__(self, *, nc: int, pois: list[np.ndarray]):
        self._nc = nc
        self._pois = [list(sorted(x)) for x in pois]
        self._init = False
        self._done = False
        self._solved = False

    def fit_u(self, traces, x):
        """Update the LDA estimates with new training data.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape `(n, nv)` and
            must be `uint16`.
        """
        if self._done:
            raise ValueError("Cannot fit_u after calling .solve(..., done=True).")
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self._acc = _scalib_ext.MultiLdaAcc(self._ns, self._nc, self._pois)
        with scalib.utils.interruptible():
            self._acc.fit(traces, x, get_config())
        self._solved = False

    def solve(self, p: int, done: bool = True):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\mathbf{W}`, the means :math:`\mathbf{\mu}_x` and the covariance
        :math:`\mathbf{\Sigma}`.

        Parameters
        ----------
        p :
            Number of dimensions to keep after dimensionality reduction for each variable.
        done :
            If True, discards intermediate state to save RAM, which prevents
            later calls to `.fit_u`.

        Notes
        -----
        Once this has been called, predictions can be performed.
        """
        if not self._init:
            raise ValueError("Cannot .solve since .fit_u was never called.")
        if self._solved:
            raise ValueError(
                "Already called .solve() on this object, should not be called twice."
            )
        with scalib.utils.interruptible():
            self._multi_lda = self._acc.multi_lda(p, get_config())

        self._solved = True
        self._done = done

        if done:
            del self._acc

    def _ldas(self, p):
        return MultiLDA.from_ldas(
            self._pois,
            [
                LDAClassifier._from_inner_solved(lda)
                for lda in self._acc.ldas(p, get_config())
            ],
        )

    def predict_proba(self, traces):
        r"""Computes the probability for each of the classes for the traces,
            for all variables.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. The array must be of dimension `(n,ns)`.

        Returns
        -------
        list of array_like, f64
            Probability distributions. Shape `(nv, n, nc)`.
        """
        if not self._solved:
            raise ValueError(
                "Call Lda.solve() before Lda.predict_proba() to compute the model."
            )
        with scalib.utils.interruptible():
            return self._multi_lda.predict_proba(traces, get_config())

    def get_sw(self):
        r"""Return :math:`S_{W}` matrix (within-class scatter)."""
        return self._acc.get_sw()

    def get_sb(self):
        r"""Return :math:`S_{B}` matrix (between-class scatter)."""
        return self._acc.get_sb()

    def get_mus(self):
        r"""Return means matrix (classes means). Shape: ``(nc, ns)``."""
        return self._acc.get_mus()
