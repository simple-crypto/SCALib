import numpy as np

def information(prs,k,priors=None):
    """
        computes the information from the probabilities. depending on how
        these are built, this function can be used to compute
            - HI: prs from samples used to build a model
            - PI: prs from fresh samples
            - MI: prs from the exact analytical expression of the leakage

        prs: (?,Nk) table contaning pr(l|x) for each possible x
        k: (?) the value corresponding to the leakage
        priors: (?) probability in every class
    """
    Nk = len(np.unique(k))
    if priors is None:
        priors = np.ones(Nk)/Nk
    entropy = -np.sum(priors*np.log2(priors))

    for x in range(Nk):
        I = np.where(k==x)[0]
        lprs = np.log2((priors[x] * prs[I,x].T/np.sum(priors * prs[I,:],axis=1)).T)
        entropy += priors[x] * np.mean(lprs)

    return entropy
