from stella.evaluation.ttest import Ttest
import numpy as np
import re
# HW setting
l = 25000
n = 2000
D = 5


traces = np.random.normal(0,10,(n,l)).astype(np.int16)
c = np.random.randint(0,2,n).astype(np.uint16)
traces = (traces.T + c).T.astype(np.int16)
def test_ttest_rust_4(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=4)

def test_ttest_rust_8(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=8)

def test_ttest_rust_16(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=16)

def test_ttest_rust_24(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=24)

def test_ttest_rust_32(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=32)

def test_ttest_rust_48(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=48)

def test_ttest_rust_1(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=True,nchunks=1)
#def test_ttest_noRUST(benchmark):
#    ttest = Ttest(l,D=D)
#    benchmark(ttest.fit_u,traces,c,use_rust=False)
