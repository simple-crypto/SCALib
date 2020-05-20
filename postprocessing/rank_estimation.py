import numpy as np
import ctypes
from ctypes import POINTER
import os
def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

def rank_estimation(logproba,init_key,nb_bins,merge_value=2,DIR=None):
    """
        estimates the rank of the correct key using hel_lib
        https://eprint.iacr.org/2016/571.pdf published at CHES2016

        logproba: (16,256) table containing the log proba in double
        init_key: 
    """
    
    Ns,Nl = logproba.shape
    if Ns != 16 or Nl != 256:
        raise Exception("Does not accept yet these shapes")

    if DIR is None:
        DIR = os.path.dirname(__file__)+"/../lib/"
    lib = ctypes.CDLL(DIR+"./hellib.so")
    run_hellib = wrap_function(lib,"stella_wrapper",None,[ctypes.c_int64,
                        ctypes.c_int32,
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_int32),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double)])

    logproba = logproba.astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    nb_bins = ctypes.c_int32(nb_bins)
    merge_value = ctypes.c_int64(merge_value)
    init_key = init_key.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    rank_min = np.double([0])
    rank_max = np.double([0])
    rank_rounded = np.double([0])


    run_hellib(merge_value,
                nb_bins,
                logproba,
                init_key,
                rank_rounded.ctypes.data_as(POINTER(ctypes.c_double)),
                rank_min.ctypes.data_as(POINTER(ctypes.c_double)),
                rank_max.ctypes.data_as(POINTER(ctypes.c_double)))

    return np.log2(rank_rounded),np.log2(rank_min),np.log2(rank_max)

if __name__ == "__main__":
    logproba = np.random.normal(0,1,(16,256)).astype(np.double)
    init_key = np.random.randint(0,256,16).astype(np.int)

    r,mi,ma = rank_estimation(logproba,init_key,nb_bins=4096,merge_value=1)

