import numpy as np
import stella.lib.rust_stella as rust
import scipy.linalg

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
        x_f64 = np.zeros(x.shape)
        
        # get sb,sw,c_means and x_f64
        rust.lda_matrix(x,y,sb,sw,c_means,x_f64,nc)
        
        # generate the projection 
        evals, evecs = scipy.linalg.eigh(sb, sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        projection = evecs[:,:self._n_components]

        # get means and cov in subspace
        means = (projection.T @ c_means.T).T 
        traces_t = (projection.T @ x_f64.T).T
        if self._n_components == 1:
            cov = np.array([[np.cov(traces_t.T)]])
        else:
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
        del x_f64,c_means,sw,sb,s,u,traces_t
        del evals,evecs,psigma_diag,above_cutoff

    def predict_proba(self,x):
        nt,_ = x.shape
        prs = np.zeros((nt,self._nc))
        rust.predict_proba_lda(x,self._projection,self._means,self._psd,prs);
        return prs;

if __name__ == "__main__":
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
    import time
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
