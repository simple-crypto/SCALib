import numpy as np

def fwht(a):
    h = 1
    x = np.zeros(a[0].shape).astype(np.float64)
    y = np.zeros(a[0].shape).astype(np.float64)
    while h < len(a):
        for i in range(0,len(a),h*2):
            for j in range(i,i+h):
                x[:] = a[j]
                y[:] = a[j+h]
                a[j] = x+y
                a[j+h] = x-y
        h *= 2

def recombine_fwht(pr):
    """
        pr is of size (Nk x D x  Nt):
            Nk size of the field
            D number of shares 
            Nt  number of traces
    """
    pr = pr.astype(np.float64)
    pr_fft = pr.copy()
    fwht(pr_fft)
    pr = np.prod(pr_fft,axis=1)
    fwht(pr)
    return pr

def information(prs,k,priors=None):
    """
        computes the information from the probabilities. depending on how
        these are built, this function can be used to compute
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
