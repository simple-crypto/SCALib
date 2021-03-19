import numpy as np
import stella.lib.rust_stella as rust
import scipy.linalg

class LDAClassifier():
    def __init__(self,x,y,nc, n_components):

        self._n_components = n_components;
        self._nc = nc
        self.lda = rust.LDA(nc,n_components)
        self.lda.fit(x,y)
        assert n_components < nc

    def predict_proba(self,x):
        prs = self.lda.predict_proba(x)
        return prs

    def __getstate__(self):
        lda = self.lda
        dic = {"means":lda.get_means(),"cov":lda.get_cov(),
                "projection":lda.get_projection(),
                "psd":lda.get_psd(),"nc":self._nc,
                "n_components":self._n_components}
        return dic
    def __setstate__(self,state):
        self.lda = rust.LDA(state["nc"],state["n_components"])
        self.lda.set_state(state["cov"],state["psd"],
                    state["means"],state["projection"],
                    state["nc"],state["n_components"])

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
    lda = LDAClassifier(traces,labels,256,n_components)
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

