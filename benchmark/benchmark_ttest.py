from stella.evaluation.ttest import Ttest
import numpy as np
import re
l = 50000
n = 5000
D = 1


traces = np.random.normal(0,10,(n,l)).astype(np.int16)
c = np.random.randint(0,2,n).astype(np.uint16)
traces = (traces.T + c).T.astype(np.int16)
def test_ttest_RUST(benchmark):

    ttest = Ttest(l,D=D)

    benchmark(ttest.fit_u,traces,c,use_rust=True)
def test_ttest_noRUST(benchmark):
    ttest = Ttest(l,D=D)
    benchmark(ttest.fit_u,traces,c,use_rust=False)
