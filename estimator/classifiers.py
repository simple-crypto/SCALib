import numpy as np
import scipy.stats 
class MultivariateGaussianClassifier():
    def __init__(self,Nc,means,covs,priors=None,dim_reduce=None):
        """
            Performs a Ns-multivariate classification on Nc classes.
            Each class is given with a mean and a covirance matrix. The function
            predict_proba is used to return the probality given leakage
            samples and the fitted model.

            Nc: number of classes
            means: (Nc,Ns) the means of each class
            covs: (Nc,Ns,Ns) the covariance of each class
            priors: (Nc) priors of each of the classes
            dim_reduce: an object that implements transform(). It is apply
                to fresh traces to reduce their dimensions.
        """
        #input checks
        if means.ndim != 2:
            raise Exception("Waiting 2 dim array for means")
        mx,my = means.shape

        if covs.ndim != 3:
            raise Exception("Waiting 3 dim array for covs")
        cx,cy,cz
        if cy != cz:
            raise Exception("Covariance matrices are not square")
        if cy != my:
            raise Exception("Missmatch cov and mean size {} vs {}".format(cy,my))
        if cx != Nc or mx != Nc:
            raise Exception("Number of class does not match the templates size")
        if priors is not None:
            if priors.ndim != 1:
                raise Exception("Waiting 1 dim array for priors")
        else:
            priors = np.ones(Nc)


        self._priors = priors
        self._means = means
        self._covs = covs
        self._Nc = Nc
        self._dim_reduce = dim_reduce
        self._l = len(mx)
    def predict_proba(self,X):
        """
            Returns the probability of each classes by applying 
            Bayes law.

            X (n_traces,Ns): n_traces traces to evaluate

            returns a (n_traces,Nc) array
        """
        if X.ndim != 2:
            raise Exception("Waiting a 2 dim array as X")
        if self._dim_reduce is not None:
            X = self._dim_reduce.transform(X)

        n_samples,Ns = X.shape
        if Ns != self._Ns:
            raise Exception("Traces do not have the expected lenght {} waiting {}".format(ny,self._l))

        prs = np.zeros((n_samples,self._Nc))

        for i in range(self._Nc):
            prs[:,i] = scipy.stats.multivariate_normal.pdf(X,
                        mean=self._means[i],cov=self._covs[i])

        I = np.where(np.sum(prs,axis=1)==0)[0]
        prs[I] = 1
        return (prs.T/np.sum(prs,axis=1)).T
