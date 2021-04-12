import numpy as np
from scaffe import _scaffe_ext

class LDAClassifier():
    r"""Models the leakage :math:`\bm{l}` with :math:`n_s` dimensions with a
    Gaussian distribution in a linear subspace with `p` dimensions. The
    projection is defined by a matrix :math:`\bm{W}` of size
    (:math:`p`, :math:`n_s`). `LDAClassifier` uses the conditional probability
    density function of the form

    .. math:: 
            \mathsf{\hat{f}}(\bm{l} | x) = 
                \frac{1}{\sqrt{(2\pi)^{p} \cdot |\bm{\Sigma} |}} \cdot
                \exp^{\frac{1}{2} 
                    (\bm{W} \cdot \bm{l} - \bm{\mu}_x)
                    \bm{\Sigma} 
                    ( \bm{W} \cdot \bm{l}-\bm{\mu}_x)'}

    where :math:`\bm{\mu}_x` is the mean of the leakage for class :math:`x`
    in the linear subspace and :math:`\bm{\Sigma}` its covariance. It
    provides the probability of each class with `predict_proba` thanks to
    Bayes' law such that

    .. math::
        \hat{\mathsf{pr}}(x|\bm{l}) = \frac{\hat{\mathsf{f}}(\bm{l}|x)}
                    {\sum_{x^*=0}^{n_c-1} \hat{\mathsf{f}}(\bm{l}|x^*)}
    
    Examples
    --------
    >>> from scaffe.modeling import LDAClassifier
    >>> import numpy as np
    >>> x = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> y = np.random.randint(0,256,5000,dtype=np.uint16)
    >>> lda = LDAClassifier(256,8,10)
    >>> lda.fit(x,y)
    >>> x = np.random.randint(0,256,(20,10),dtype=np.int16)
    >>> predicted_proba = lda.predict_proba(x)

    Notes
    ----- 
    This implementation uses custom implementation of
    `sklearn.LDA(solver="eigen")` to compute the projection matrix and a custom
    implementation of `scipy.stats.multivariate_normal.pdf()`.

    [1] François-Xavier Standaert and Cédric Archambeau, "Using
    Subspace-Based Template Attacks to Compare and Combine Power and
    Electromagnetic Information Leakages", CHES 2008: 411-425

    Parameters
    ----------
    nc : int
        Number of possible classes (e.g., 256 for 8-bit target). `nc` must
        be smaller than `65536`.
    p : int
        Number of dimensions in the linear subspace.
    ns: int
        Number of dimensions in the leakage.
    """
    def __init__(self,nc, p,ns):
        self.p_ = p;
        self.nc_ = nc
        self.ns_ = ns
        self.lda = _scaffe_ext.LDA(nc,p,ns)
        assert p < nc

    def fit(self,l,x):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\bm{W}`, the means :math:`\bm{\mu}_x` and the covariance
        :math:`\bm{\Sigma}`.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint16
            Labels for each trace. Must be of shape `(n)` and
            must be `uint16`.
        """
        self.lda.fit(l,x)

    def predict_proba(self,l):
        r"""Computes the probability for each of the classes for the traces
        contained in `x`.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.

        Returns
        -------
        prs : array_like, f64
            Array to contains probabilities on each of its rows. `prs` is of
            shapre `(n,ns)`.
        """
        prs = self.lda.predict_proba(l)
        return prs

    def __getstate__(self):
        lda = self.lda
        dic = {"means":lda.get_means(),"cov":lda.get_cov(),
                "projection":lda.get_projection(),
                "psd":lda.get_psd(),"nc":self.nc_,
                "p":self.p_,"ns":self.ns_}
        return dic
    
    def __setstate__(self,state):
        self.lda = _scaffe_ext.LDA(state["nc"],state["p"],state["ns"])
        self.lda.set_state(state["cov"],state["psd"],
                    state["means"],state["projection"],
                    state["nc"],state["p"],state["ns"])
        self.nc_ = state["nc"]
        self.ns_ = state["ns"]
        self.p_ = state["p"]
