import pytest
from scaffe.metrics import Ttest
import numpy as np
import scipy.stats

def test_ttest_d1():
    ns = 10
    d = 1
    nc = 2

    n = 200
    
    m = np.random.randint(0,2,(nc,ns))
    traces = np.random.randint(0,10,(n,ns),dtype=np.int16)
    labels = np.random.randint(0,nc,n,dtype=np.uint16)
    traces += m[labels] 

    I0 = np.where(labels == 0)[0]
    n0 = len(I0)
    u0 = np.mean(traces[I0,:],axis=0)
    v0 = np.var(traces[I0,:],axis=0)

    I1 = np.where(labels == 1)[0]
    n1 = len(I1)
    u1 = np.mean(traces[I1,:],axis=0)
    v1 = np.var(traces[I1,:],axis=0)

    t_ref = (u0 - u1)/np.sqrt((v1/n1) + (v0/n0))


    ttest = Ttest(ns,1)
    ttest.fit_u(traces,labels)
    t = ttest.get_ttest()
    assert(np.allclose(t_ref,t,rtol=1E-3))


def test_ttest_d2():
    ns = 10
    d = 2
    nc = 2

    n = 200
    
    m = np.random.randint(0,2,(nc,ns))
    traces = np.random.randint(0,10,(n,ns),dtype=np.int16)
    labels = np.random.randint(0,nc,n,dtype=np.uint16)
    traces += m[labels] 

    I0 = np.where(labels == 0)[0]
    n0 = len(I0)
    u0 = np.mean(traces[I0,:],axis=0)
    s0 = np.std(traces[I0,:],axis=0)
    u_0 = np.mean((traces[I0,:] - u0)**2,axis=0)
    s2_0 = np.mean((traces[I0,:] - u0)**4,axis=0)
    s2_0 -= np.mean((traces[I0,:] - u0)**2,axis=0)**2

    I1 = np.where(labels == 1)[0]
    n1 = len(I1)
    u1 = np.mean(traces[I1,:],axis=0)
    s0 = np.std(traces[I1,:],axis=0)
    u_1 = np.mean((traces[I1,:] - u1)**2,axis=0)
    s2_1 = np.mean((traces[I1,:] - u1)**4,axis=0)
    s2_1 -= np.mean((traces[I1,:] - u1)**2,axis=0)**2

    t_ref = (u_0 - u_1)/np.sqrt((s2_1/n1) + (s2_0/n0))
    
    ttest = Ttest(ns,d)
    ttest.fit_u(traces,labels)
    t = ttest.get_ttest()[d-1,:]
    assert(np.allclose(t_ref,t,rtol=1E-3))
