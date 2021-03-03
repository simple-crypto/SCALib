import numpy as np
import stella.lib.rust_stella as rust
AND = 0
XOR = 1
XOR_CST = 2
symbols = {"&":{"val":AND,"inputs_distri":2},
        "^":{"val":XOR,"inputs_distri":2},
        "+":{"val":XOR_CST,"inputs_distri":1}}

delimiter = "#indeploop"
end_delimiter = "#endindeploop"
secret_flag = "#secret"
public_flag = "#public"
profile_flag = "#profile"

secret_flag_v = 1
public_flag_v = 2
profile_flag_v = 4

CLIP = 1E-50

def new_variable(i):
    return {"id":i,"neighboors":[],"flags":0,"in_loop":False}
def new_function(i,func):
    return {"id":i,"inputs":[],"outputs":[],"func":func}

def init_graph_memory(functions,variables,N,Nc):
    # init the distribution
    for var in variables:
        var = variables[var]
        in_loop = var["in_loop"]
        if in_loop: 
            n = N 
        else: 
            n = 1

        if var["flags"] & public_flag_v != 0:
            var["values"] = np.zeros(n,dtype=np.uint32)
        else:
            if var["flags"] & profile_flag_v != 0:
                var["distri_orig"] = np.ones((n,Nc))
            var["distri"] = np.zeros((n,Nc))
            var["msg"] = np.zeros((N,len(var["neighboors"]),Nc))

    variables_list = list(map(lambda x:variables[x],variables))
    for func in functions:
        neighboors = func["neighboors"]
        func["msg"] = np.zeros((N,len(neighboors),Nc))

    return functions,variables_list,variables
 
def create_graph(fname):
    functions = []
    variables = {}
    in_loop = False
    with open(fname) as fp:
        lines = map(lambda l:l.rstrip('\n'),fp.readlines())
        lines = filter(lambda l : len(l)>0 and l[0] != '%',lines)

    # for each line
    for line in lines:
        split = line.split(" ")
        # delimiter
        if delimiter in split:
            in_loop = True
            continue
        if end_delimiter in split:
            in_loop = False
            continue

        v = split[0]

        # create variable if not exists
        if v in variables:
            node = variables[v]
        else:
            node = new_variable(len(variables))
            node["in_loop"] = in_loop
            variables[v] = node
    
        # add the flags
        if secret_flag in split:
            node["flags"] |= secret_flag_v
        if public_flag in split:
            node["flags"] |= public_flag_v 
        if profile_flag in split:
            node["flags"] |= profile_flag_v

        # add function
        op = list(set(split) & set(list(symbols)))
        if len(op) > 0:
            assert in_loop
            i = len(functions)
            v = variables[split[0]] 
            a = variables[split[2]]
            b = variables[split[4]]
            a["neighboors"].append(i)
            b["neighboors"].append(i)
            v["neighboors"].append(i)

            func = new_function(i,symbols[op[0]]["val"])
            func["in_loop"] = in_loop
            func["inputs"].append(a["id"])
            func["inputs"].append(b["id"])
            func["outputs"].append(v["id"])
            func["neighboors"] = func["inputs"] + func["outputs"]
            functions.append(func)

    # init the distribution
    for var in variables:
        var = variables[var]
        if var["flags"] & public_flag_v != 0:
            continue
        else:
            var["offset"] = [None for i in var["neighboors"]]
            for i,neighboor in enumerate(var["neighboors"]):
                neighboor = functions[neighboor]
                var["offset"][i] = neighboor["neighboors"].index(var["id"])
    
    variables_list = list(map(lambda x:variables[x],variables))
    for func in functions:
        neighboors = func["neighboors"]
        func["offset"] = [None for i in neighboors]
        for i,neighboor in enumerate(neighboors):
            neighboor = variables_list[neighboor]
            func["offset"][i] = neighboor["neighboors"].index(func["id"])

    return functions,variables_list,variables

def reset_graph_memory(variables_list,Nc):

    for var in variables_list:
        if var["flags"] & public_flag_v != 0:
            continue

        if "distri_orig" in var:
            np.clip(var["distri_orig"],CLIP,1,out=var["distri_orig"])
            var["distri_orig"][:,:]= (var["distri_orig"].T / np.sum(var["distri_orig"],axis=1)).T
            for i in range(len(var["msg"][0,:,0])):
                var["msg"][:,i,:] = var["distri_orig"]
        else:
            var["msg"][:] = 1/Nc
        np.clip(var["msg"],CLIP,1,out=var["msg"])

if __name__ == "__main__":
    functions,variables_list,variables = create_graph("example_graph.txt")
    n = 20000
   
    from tqdm import tqdm
    for nc in 2**np.arange(7,8):
        init_graph_memory(functions,variables,n,nc)
        for it in tqdm(range(1),desc="nc %d"%(nc)):
            x_0 = np.random.randint(0,nc)
            p_0 = np.random.randint(0,nc)
            x_1 = np.random.randint(0,nc)
            p_1 = np.random.randint(0,nc)

            k_0_expected = p_0 ^ x_0
            k_1_expected = p_1 ^ x_1
            k_2_expected = k_1_expected ^ k_0_expected
            k_3_expected = p_0 ^ x_0

            preci = (np.random.random(n)*(1 - 1/nc)).reshape(n,1) + 1/nc
            variables["p_0"]["distri_orig"][:,:] = (1-preci)/(nc-1)
            variables["p_0"]["distri_orig"][:,p_0] = preci[:,0]

            preci = (np.random.random(n)*(1 - 1/nc)).reshape(n,1) + 1/nc
            variables["x_0"]["distri_orig"][:,:] = (1-preci)/(nc-1)
            variables["x_0"]["distri_orig"][:,x_0] = preci[:,0]

            preci = (np.random.random(n)*( (1 - 1/nc) + 1/nc)).reshape(n,1)
            preci = (np.random.random(n)*(1 - 1/nc)).reshape(n,1) + 1/nc
            variables["p_1"]["values"][:] = p_1
            variables["x_1"]["distri_orig"][:,:] = (1-preci)/(nc-1)
            variables["x_1"]["distri_orig"][:,x_1] = preci[:,0]

            reset_graph_memory(variables_list,nc)
            for i in range(2):
                rust.belief_propagation(functions,variables_list)
            k_0 = np.argmax(variables["k_0"]["distri"],axis=1)[0]
            
            k_1 = np.argmax(variables["k_1"]["distri"],axis=1)[0]

            k_2 = np.argmax(variables["k_2"]["distri"],axis=1)[0]

            k_3 = np.argmax(variables["k_3"]["distri"],axis=1)[0]

            assert k_0 == k_0_expected 
            assert k_1 == k_1_expected 
            assert k_2 == k_2_expected 
            #assert k_3 == k_3_expected 
