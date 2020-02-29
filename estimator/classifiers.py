import numpy as np
import scipy.stats 
class MultivariateGaussianClassifier():
    def __init__(self,Nc,means,covs,priors=None,dim_reduce=None):
        """
            Performs a Ns-multivairate classification on Nc classes 

            means: (Nc,Ns) the means of each class
            covs: (Nc,Ns,Ns) the covariance of each class
            priors: (Nc) priors of each of the classes
        """

        if priors is None:
            priors = np.ones(Nc)
        self._priors = priors
        self._means = means
        self._covs = covs
        self._Nc = Nc
        self._dim_reduce = dim_reduce

    def predict_proba(self,X):
        """
            Predict the probability for each of the classes given a
            sample

            X (?,Ns): ? Ns dimensional samples
        """
        if self._dim_reduce is not None:
            X = self._dim_reduce.transform(X)
        n_samples,Ns = X.shape

        prs = np.zeros((n_samples,self._Nc))

        for i in range(self._Nc):
            prs[:,i] = scipy.stats.multivariate_normal.pdf(X,
                        mean=self._means[i],cov=self._covs[i])

        I = np.where(np.sum(prs,axis=1)==0)[0]
        prs[I] = 1
        return (prs.T/np.sum(prs,axis=1)).T
