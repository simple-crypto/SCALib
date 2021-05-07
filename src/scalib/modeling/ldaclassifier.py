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
    >>> lda.fit(x,y)
    >>> x = np.random.randint(0,256,(20,10),dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(x)

    Notes
    -----
    This implementation uses custom implementation of
    `sklearn.LDA(solver="eigen")` to compute the projection matrix and a custom
    implementation of `scipy.stats.multivariate_normal.pdf()`.

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
        self.p_ = p
        self.nc_ = nc
        self.ns_ = ns
        self.lda = _scalib_ext.LDA(nc, p, ns)
        assert p < nc

    def fit(self, l, x):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\mathbf{W}`, the means :math:`\mathbf{\mu}_x` and the covariance
        :math:`\mathbf{\Sigma}`.


        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape `(n)` and
            must be `uint16`.

        Notes
        -----
        This method does not support updating the model: calling this method
        twice overrides the previous result.
        """
        self.lda.fit(l, x)

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
        prs = self.lda.predict_proba(l)
        return prs

    def __getstate__(self):
        lda = self.lda
        dic = {
            "means": lda.get_means(),
            "cov": lda.get_cov(),
            "projection": lda.get_projection(),
            "psd": lda.get_psd(),
            "nc": self.nc_,
            "p": self.p_,
            "ns": self.ns_,
        }
        return dic

    def __setstate__(self, state):
        self.lda = _scalib_ext.LDA(state["nc"], state["p"], state["ns"])
        self.lda.set_state(
            state["cov"],
            state["psd"],
            state["means"],
            state["projection"],
            state["nc"],
            state["p"],
            state["ns"],
        )
        self.nc_ = state["nc"]
        self.ns_ = state["ns"]
        self.p_ = state["p"]
