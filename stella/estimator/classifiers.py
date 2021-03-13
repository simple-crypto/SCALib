import numpy as np
import stella.lib.rust_stella as rust

class LDAClassifier():
    def __init__(self,nc, n_components):

        self._n_components = n_components;
        self._nc = nc
        assert n_components < nc
    
    def fit(self,x,y):
        
        # this method is inspired by the sklearn implementation of scikit-learn
        # they are equivalent but enables rust acceleration

        nt,ns = x.shape
        nc = self._nc

        # pre allocate arrays
        sw = np.zeros((ns,ns))
        sb = np.zeros((ns,ns))
        c_means = np.zeros((nc,ns))
        x_f64 = np.zeros(traces.shape)
        
        # get sb,sw,c_means and x_f64
        rust.lda_matrix(traces,labels,sb,sw,c_means,x_f64,nc)

        # generate the projection 
        evals, evecs = scipy.linalg.eigh(sb, sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        projection = evecs[:,:n_components]

        # get means and cov in subspace
        means = (projection.T @ c_means.T).T 
        traces_t = (projection.T @ x_f64.T).T
        cov = np.cov(traces_t.T)

        # generate psd for the covariance matrix 
        s,u = scipy.linalg.eigh(cov)
        t = u.dtype.char.lower()
        cond = np.max(np.abs(s)) * max(cov.shape) * np.finfo(t).eps
        above_cutoff = (abs(s) > cond)
        psigma_diag = 1.0 / s[above_cutoff]
        u = u[:, above_cutoff]
        U = np.multiply(u, np.sqrt(psigma_diag))
        psd = U
        
        self._projection = projection
        self._cov = cov
        self._means = means
        self._psd = psd
        del x_f64,c_means,sw,sb

    def predict_proba(self,x):
        nt,_ = x.shape
        prs = np.zeros((nt,self._nc))
        rust.predict_proba_lda(x,self._projection,self._means,self._psd,prs);
        return prs;
#####################
#####################
# This comes from SciPy
#####################
#####################
def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """
    Determine which eigenvalues are "small" given the spectrum.
    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.
    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.
    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.
    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.
    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
    """
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)


class _PSD(object):
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.
    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().
    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)
    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().
    """

    def __init__(self, M, cond=None, rcond=None, lower=True,
                 check_finite=True, allow_singular=True):
        # Compute the symmetric eigendecomposition.
        # Note that eigh takes care of array conversion, chkfinite,
        # and assertion that the matrix is square.
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            raise np.linalg.LinAlgError('singular matrix')
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))

        # Initialize the eagerly precomputed attributes.
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))

        # Initialize an attribute to be lazily computed.
        self._pinv = None

    @property
    def pinv(self):
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv


if __name__ == "__main__":
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
    import time
    import scipy
    ns = 20
    n_components = 10

    traces = np.random.randint(0,256,(200000,ns),dtype=np.int16)
    labels = np.random.randint(0,256,200000,dtype=np.uint16)

    start = time.time()
    lda_ref = LDA_sklearn(solver="eigen")
    lda_ref.fit(traces,labels)
    print(time.time()-start)

    start = time.time()
    lda = LDAClassifier(256,n_components)
    lda.fit(traces,labels)
    print(time.time()-start)

    # check equivalence    
    traces_t = (lda_ref.scalings_[:,:n_components].T @ traces.T).T
    means_check = np.zeros((256,n_components))
    for i in range(256):
        I = np.where(labels==i)[0]
        means_check[i,:] = np.mean(traces_t[I,:],axis=0)
    traces_t = traces_t - means_check[labels,:]
    cov_check = np.cov(traces_t.T)
   
    traces = np.random.randint(0,256,(1000,ns),dtype=np.int16)
    start = time.time()
    prs = lda.predict_proba(traces)
    print(time.time() - start)
    
    traces_t = (lda._projection[:,:n_components].T @ traces.T).T
    prs_ref = np.zeros((len(traces),256))
    for x in range(256):
        prs_ref[:,x] = scipy.stats.multivariate_normal.pdf(traces_t,lda._means[x],lda._cov)
    prs_ref = (prs_ref.T /  np.sum(prs_ref,axis=1)).T

    assert(np.allclose(prs,prs_ref))
    assert(np.allclose(lda._projection,lda_ref.scalings_[:,:n_components]))
    assert(np.allclose(means_check,lda._means))
    assert(np.allclose(cov_check,lda._cov))
    # assert(np.allclose(psd_check,lda._psd))
