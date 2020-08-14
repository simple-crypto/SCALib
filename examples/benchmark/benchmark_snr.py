from stella.evaluation.snr import SNR
import numpy as np
import re

Nc = 16
Np = 8
l = 50000
n = 1000
def test_SNR_RUST(benchmark):
    traces = np.random.randint(0,256,(n,l),dtype=np.int16)
    x = np.random.randint(0,Nc,(Np,l),dtype=np.int16)
    snr = SNR(Nc,l,Np)
    benchmark(snr.fit_u,traces,x,use_rust=True)
def test_SNR_noRUST(benchmark):
    traces = np.random.randint(0,256,(n,l),dtype=np.int16)
    x = np.random.randint(0,Nc,(Np,l),dtype=np.int16)
    snr = SNR(Nc,l,Np)
    benchmark(snr.fit_u,traces,x,use_rust=False)
if __name__ == "__main__":
    traces = np.random.randint(0,256,(n,l),dtype=np.int16)
    x = np.random.randint(0,Nc,(Np,l),dtype=np.int16)
    snr = SNR(Nc,l,Np)
    snr.fit_u(traces,x,use_rust=True)
