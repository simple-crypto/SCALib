import numpy as np
class PCA:
    def __init__(self,means,M):
        """ means of size NkxN
                    M is the number of dimensions to keep
        """
        self._means = means
        self._M = M
        Nk = len(means[:,0])

        tk = np.mean(means,axis=0)
        T = (means-tk).T # of size NxNk
        self._T = T

        D,U = np.linalg.eig(np.dot((1/Nk)*T.T,T)) # D value, U, vectors
        perm = np.flip(np.argsort(D))
        D = D[perm]
        U = U[:,perm]
        D = np.abs(np.diag(D))
        self._D = D
        V = np.sqrt(1/Nk) * np.dot(np.dot(T,U),(np.linalg.inv(D**(1/2)))) # Normalized vectors

        self._V = V
        self._D = D

    def transform(self,traces):
        W = self._V[:,:self._M]
        return np.dot(W.T,traces.T).T

