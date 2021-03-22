import numpy as np
import stella.lib.stella as rust
import scipy.linalg

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
    >>> from stella.modeling import LDAClassifier
    >>> import numpy as np
    >>> x = np.random.randint(0,256,(50000,200),dtype=np.int16)
    >>> y = np.random.randint(0,256,50000,dtype=np.uint16)
    >>> lda = LDAClassifier(256,8)
    >>> lda.fit(x,y)
    >>> x = np.random.randint(0,256,(50000,200),dtype=np.int16)
    >>> lda.predict_proba(x)

    Notes
    ----- 
    This implementation uses custom implementation of.
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
    """
    def __init__(self,nc, p):
        self.p_ = p;
        self.nc_ = nc
        self.lda = rust.LDA(nc,p)
        assert p < nc

    def fit(self,x,y):
        r"""Estimates the PDF parameters that is the projection matrix
        :math:`\bm{W}`, the means :math:`\bm{\mu}_x` and the covariance
        :math:`\bm{\Sigma}`.

        Parameters
        ----------
        x : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        y : array_like, uint16
            Labels for each trace. Must be of shape `(n)` and
            must be `uint16`.
        """
        self.lda.fit(x,y)

    def predict_proba(self,x):
        r"""Computes the probability for each of the classes for the traces
        contained in `x`.

        Parameters
        ----------
        x : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.

        Returns
        -------
        prs : array_like, f64
            Array to contains probabilities on each of its rows. `prs` is of
            shapre `(n,ns)`.
        """
        prs = self.lda.predict_proba(x)
        return prs

    def __getstate__(self):
        lda = self.lda
        dic = {"means":lda.get_means(),"cov":lda.get_cov(),
                "projection":lda.get_projection(),
                "psd":lda.get_psd(),"nc":self.nc_,
                "p":self.p_}
        return dic
    
    def __setstate__(self,state):
        self.lda = rust.LDA(state["nc"],state["p"])
        self.lda.set_state(state["cov"],state["psd"],
                    state["means"],state["projection"],
                    state["nc"],state["p"])

if __name__ == "__main__":
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
    import time
    ns = 2000
    n_components = 2
    sb = np.random.random((ns,ns))
    sw = np.random.random((ns,ns))
    sb = sb @ sb.T
    sw = sw @ sw.T
    print("start")
    scipy.linalg.eigh(sw,sb)
    print("done")
    m = np.random.randint(0,4,(256,ns))
    traces = np.random.randint(0,10,(200000,ns),dtype=np.int16)
    labels = np.random.randint(0,256,200000,dtype=np.uint16)
    traces += m[labels] 

    start = time.time()
    lda = LDAClassifier(256,n_components)
    lda.fit(traces,labels)
    print(time.time()-start)
    
    start = time.time()
    lda_ref = LDA_sklearn(solver="eigen")
    lda_ref.fit(traces,labels)
    print(time.time()-start)


    # check equivalence    
    traces_t = (lda_ref.scalings_[:,:n_components].T @ traces.T).T
    means_check = np.zeros((256,n_components))
    for i in range(256):
        I = np.where(labels==i)[0]
        means_check[i,:] = np.mean(traces_t[I,:],axis=0)
    traces_t = traces_t - means_check[labels,:]
    cov_check = np.cov(traces_t.T)


    traces = np.random.randint(0,10,(200,ns),dtype=np.int16)
    labels = np.random.randint(0,256,200,dtype=np.uint16)
    traces += m[labels] 
     
    start = time.time()
    prs = lda.predict_proba(traces)
    print(time.time() - start)
 
    traces_t = (lda_ref.scalings_[:,:n_components].T @ traces.T).T
    prs_ref = np.zeros((len(traces),256))
    for x in range(256):
        prs_ref[:,x] = scipy.stats.multivariate_normal.pdf(traces_t,
                            means_check[x],cov_check)
    prs_ref = (prs_ref.T /  np.sum(prs_ref,axis=1)).T

    assert(np.allclose(lda.lda.get_projection(),lda_ref.scalings_[:,:n_components]))
    assert(np.allclose(means_check,lda.lda.get_means().T))
    assert(np.allclose(cov_check,lda.lda.get_cov()))
    assert(np.allclose(prs,prs_ref))

