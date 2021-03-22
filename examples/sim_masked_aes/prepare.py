from common import gen_traces_serial,sbox
import numpy as np
import pickle
from settings import * 

print_settings()
print("Generate profile traces")
gen_traces_serial(nfile_profile,
                    ntraces,std,tag,
                    DIR_PROFILE,random_key=True,D=D)

print("Generate attack traces")
secret_key = gen_traces_serial(nfile_attack,ntraces,
                        std,tag,DIR_ATTACK,
                        random_key=False,D=D)[0]

print("Generate the fgraph")
with open(fgraph, 'w') as fp:
    fp.write("sbox #table \n")
    for i in range(16): fp.write("k%d #secret \n"%(i))
    fp.write("\n\n#indeploop\n\n")
    for i in range(16): fp.write("p%d #public\n"%(i))
    for i in range(16): fp.write("y%d = k%d + p%d\n"%(i,i,i))
    for i in range(16): fp.write("x%d = y%d -> sbox \n"%(i,i))
    for i in range(16):
        for d in range(D): fp.write("x%d_%d #profile\n"%(i,d))
        for d in range(D): fp.write("y%d_%d #profile\n"%(i,d))
        add = ' ^ '.join(["x%d_%d"%(i,d) for d in range(D)])
        fp.write("x%d = "%(i)+add+"\n")
        add = ' ^ '.join(["y%d_%d"%(i,d) for d in range(D)])
        fp.write("y%d = "%(i)+add+"\n")
        fp.write("\n")
    fp.write("\n\n#endindeploop\n\n")

print("Store the secret key")
pickle.dump(secret_key,open("secret_key.pkl",'wb'))
