from stella.sasca.DataHolder import *
from stella.sasca.Node import *

def XOR(out,a,b,D):
    for d in range(D):
        out[d] = a[d] ^ b[d]

def NXOR(out,a,b,D):
    XOR(out,a,b,D)
    out[0] = ~out[0];

def AND_ISW(out,a,b,rng,D):
    for d in range(D):
        out[d] = a[d] & b[d]

    for i in range(D):
        for j in range(i+1,D):
            s = VNode(value=rng.get_row())
            tmp = (a[i] & b[j]) ^ s
            sp = tmp ^ (a[j] & b[i])
            out[i] = out[i] ^ s
            out[j] = out[j] ^ sp

