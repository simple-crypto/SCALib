import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scalib import _scalib_ext


class LDAClassifier:
    r"""Models the leakage :math:`\mathbf{l}` with :math:`n_s` dimensions using
    linear discriminant analysis dimentionality reduction and gaussian
    templates.
    Based on the training data, linear discriminant analysis build a linear
    dimentionality reduction to :math:`p` dimensions that maximizes class
    separation.
    Then, a multivariate gaussian template is fitted for each class (using the
    same covariance matrix for all the classes) in the reduced dimensionality
    space to predict leakage likelihood [1]_.

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

    `LDAClassifier` provides the probability of each class with `predict_proba`
    thanks to Bayes' law such that

    .. math::
        \hat{\mathsf{pr}}(x|\mathbf{l}) = \frac{\hat{\mathsf{f}}(\mathbf{l}|x)}
                    {\sum_{x^*=0}^{n_c-1} \hat{\mathsf{f}}(\mathbf{l}|x^*)}.

    Examples
    --------
    >>> from scalib.modeling import LDAClassifier
    >>> import numpy as np
    >>> x = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> y = np.random.randint(0,256,5000,dtype=np.uint16)
    >>> lda = LDAClassifier(256,3,10)
    >>> lda.fit_u(x,y, 0)
    >>> lda.solve()
    >>> x = np.random.randint(0,256,(20,10),dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(x)

    Notes
    -----
    This should have similar behavior as `sklearn.LDA`, but it has better
    performance (and lower flexibility).

    .. [1] François-Xavier Standaert and Cédric Archambeau, "Using
       Subspace-Based Template Attacks to Compare and Combine Power and
       Electromagnetic Information Leakages", CHES 2008: 411-425

    Parameters
    ----------
    nc : int
        Number of possible classes (e.g., 256 for 8-bit target). `nc` must
        be smaller than `2**16`.
    p : int
        Number of dimensions in the linear subspace.
    ns: int
        Number of dimensions in the leakage.
    """

    def __init__(self, nc, p, ns):
        self.solved = False
        self.p = p
        self.acc = _scalib_ext.LdaAcc(nc, ns)
        assert p < nc

    def fit_u(self, l, x, gemm_mode=1):
        r"""Update statistical model estimates with fresh data.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape `(n)` and
            must be `uint16`.
        gemm_mode: int (default 1)
            0: use matrixmultiply matrix multiplication.
            n>0: use n threads with BLIS matrix multiplication.
            BLIS is only used on linux. Matrixmultiply is always used on other
            OSes.
        """
        self.acc.fit(l, x, gemm_mode)
        self.solved = False

    def solve(self):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\mathbf{W}`, the means :math:`\mathbf{\mu}_x` and the covariance
        :math:`\mathbf{\Sigma}`.

        Notes
        -----
        Once this has been called, predictions can be performed.
        """
        self.lda = self.acc.lda(self.p)
        self.solved = True

    def predict_proba(self, l):
        r"""Computes the probability for each of the classes for the traces
        contained in `l`.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.

        Returns
        -------
        array_like, f64
            Probabilities. Shape `(n, nc)`.
        """
        assert (
            self.solved
        ), "Call LDA.solve() before LDA.predict_proba() to compute the model."
        prs = self.lda.predict_proba(l)
        return prs

    def __getstate__(self):
        dic = {
            "solved": self.solved,
            "p": self.p,
            "acc": self.acc.get_state(),
        }
        try:
            dic["lda"] = self.lda.get_state()
        except AttributeError:
            pass
        return dic

    def __setstate__(self, state):
        self.solved = state["solved"]
        self.p = state["p"]
        self.acc = _scalib_ext.LdaAcc.from_state(*state["acc"])
        if "lda" in state:
            self.lda = _scalib_ext.LDA.from_state(*state["lda"])


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
        variable `i`, and training trace `t`, `t[pois[i]]` is the input
        datapoints for the LDA.
    num_cpus: int or None (default)
        Total number of threads to use in parallel (taking into
        account multi-threading in a single LDA parameterized with gemm_mode).
        If None, takes the number of available logial CPUs.
    gemm_mode: int
        0: use matrixmultiply matrix multiplication.
        n>0: use n threads with BLIS matrix multiplication.
        BLIS is only used on linux. Matrixmultiply is always used on other
        OSes.

    Examples
    --------
    >>> from scalib.modeling import MultiLDA
    >>> import numpy as np
    >>> # 5000 traces with 50 points each
    >>> x = np.random.randint(0, 256, (5000,50),dtype=np.int16)
    >>> # 5 variables (8-bit), and 5000 traces
    >>> y = np.random.randint(0, 256, (5000, 5),dtype=np.uint16)
    >>> # 10 POIs for each of the 5 variables
    >>> pois = [list(range(7*i, 7*i+10)) for i in range(5)]
    >>> # Keep 3 dimensions after dimensionality reduction
    >>> lda = MultiLDA(5*[256], 5*[3], pois)
    >>> lda.fit_u(x, y)
    >>> lda.solve()
    >>> # Predict the class for 20 traces.
    >>> x = np.random.randint(0, 256, (20, 50), dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(x)
    """

    def __init__(self, ncs, ps, pois, num_cpus=None, gemm_mode=4):
        self.pois = pois
        self.num_cpus = num_cpus
        if self.num_cpus is None:
            try:
                self.num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_cpus = os.cpu_count()
                if self.num_cpus is None:
                    # default to one thread
                    self.num_cpus = 1
        if gemm_mode == 0:
            self.num_threads = self.num_cpus
        else:
            self.num_threads = max(1, self.num_cpus // gemm_mode)
        self.gemm_mode = gemm_mode
        self.ldas = [
            LDAClassifier(nc, p, len(poi)) for nc, p, poi in zip(ncs, ps, pois)
        ]

    def fit_u(self, l, x):
        """Update the LDA estimates with new training data.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape `(n, nv)` and
            must be `uint16`.
        """
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(
                executor.map(
                    lambda i: self.ldas[i].fit_u(
                        l[:, self.pois[i]], x[:, i], self.gemm_mode
                    ),
                    range(len(self.ldas)),
                )
            )

    def solve(self):
        """See `LDAClassifier.solve`."""
        # Use num_cpus instead of num_threads as there is not parallelism
        # inside of lda.solve.
        with ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
            list(executor.map(lambda lda: lda.solve(), self.ldas))

    def predict_proba(self, l):
        """Predict probabilities for all variables.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)`.

        Returns
        -------
        list of array_like, f64
            Probabilities. `nv` arrays of shape `(n, nc)`.
        See `LDAClassifier.solve`.
        """
        # Use num_cpus instead of num_threads as there is not parallelism
        # inside of lda.predict_proba.
        with ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
            return executor.map(
                lambda i: self.ldas[i].predict_proba(l[:, self.pois[i]]),
                range(len(self.ldas)),
            )
