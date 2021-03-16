import numpy as np
import stella.lib.rust_stella as rust
AND = 0
XOR = 1
XOR_CST = 2
LOOKUP = 3
symbols = {"&":{"val":AND,"inputs_distri":2},
        "^":{"val":XOR,"inputs_distri":-1},
        "+":{"val":XOR_CST,"inputs_distri":1},
        "->":{"val":LOOKUP,"inputs_distri":1}}

delimiter = "#indeploop"
end_delimiter = "#endindeploop"
secret_flag = "#secret"
public_flag = "#public"
profile_flag = "#profile"
tab_flag = "#table"

SECRET = 1
PUBLIC = 2
PROFILE = 4
TABLE = 8

CLIP = 1E-50

def new_variable(i):
    return {"id":i,"neighboors":[],"flags":0,"in_loop":False}
def new_function(i,func):
    return {"id":i,"inputs":[],"outputs":[],"func":func}

def init_graph_memory(graph,N,Nc):
    functions = graph["functions"]
    variables = graph["var"]
    # init the distribution
    for var in variables:
        var = variables[var]
        in_loop = var["in_loop"]
        if in_loop: 
            n = N 
        else: 
            n = 1

        if var["flags"] & (PUBLIC) != 0:
            if "values" in var: del var["values"]
            var["values"] = np.zeros(n,dtype=np.uint32)
        elif var["flags"] & (TABLE) != 0:
            if "table" in var: del var["table"]
            var["table"] = np.zeros(Nc,dtype=np.uint32)
        else:
            if var["flags"] & PROFILE != 0:
                if "distri_orig" in var: del var["distri_orig"]
                var["distri_orig"] = np.ones((n,Nc))
            if "distri" in var: del var["distri"]
            var["distri"] = np.zeros((n,Nc))

    for p in graph["publics"]:
        graph["publics"][p] = np.zeros(N,dtype=np.uint32)
    for p in graph["tables"]:
        graph["tables"][p] = np.zeros(Nc,dtype=np.uint32)
 
def create_graph(fname):
    functions = []
    variables = {}
    publics = {}
    tables = {}
    vertex = 0
    in_loop = False
    with open(fname) as fp:
        lines = map(lambda l:l.rstrip('\n'),fp.readlines())
        lines = filter(lambda l : len(l)>0 and l[0] != '%',lines)

    # for each line
    for line in lines:
        split = line.split()
        # delimiter
        if delimiter in split:
            in_loop = True
            continue
        if end_delimiter in split:
            in_loop = False
            continue

        # get current variable
        v = split[0]

        # create variable if not exists
        if v in variables:
            node = variables[v]
        else:
            node = new_variable(len(variables))
            node["in_loop"] = in_loop
        
        insert = True
        # add the flags
        if secret_flag in split:
            node["flags"] |= SECRET 
        if public_flag in split:
            node["flags"] |= PUBLIC; 
            insert = False; publics[v] = []
        if profile_flag in split:
            node["flags"] |= PROFILE
        if tab_flag in split:
            node["flags"] |= TABLE;
            insert = False; tables[v] = []
        
        if insert:
            variables[v] = node

        # add function if line contains one symbol
        op = list(set(split) & set(list(symbols)))
        if len(op) > 0:
            # operation are only allowed in the loop
            assert in_loop

            # get the function sumbol and id and output
            f = symbols[op[0]]
            i = len(functions)
            v = variables[split[0]] 

            # create new function
            func = new_function(i,f["val"])
            func["in_loop"] = in_loop

            # add relation between fct and output
            v["neighboors"].append(vertex); 
            func["outputs"].append(vertex); vertex+=1

            # add relation between fct and inputs
            for j,labels in enumerate(split[::2][1:]):
                # add neighboors only if the input has a distribution
                if f["inputs_distri"] == -1 or j < f["inputs_distri"]:
                    v = variables[labels]
                    func["inputs"].append(vertex)
                    v["neighboors"].append(vertex); vertex+=1
          
            if f["val"] == LOOKUP:
                func["table_label"] = split[-1]
            elif f["val"] == XOR_CST:
                func["value_label"] = split[-1]

            # set as neighboors all the inputs that have a distribution
            func["neighboors"] = func["outputs"].copy()
            func["neighboors"] += func["inputs"][:].copy() 
            functions.append(func)

    # generate the list
    variables_list = list(map(lambda x:variables[x],variables))
   
    return {"functions":functions,"var_list":variables_list,"vertex":vertex,
                    "var":variables,"publics":publics,"tables":tables}

def run_bp(graph,it,ntraces,nc):
    reset_graph_memory(graph,nc)
    rust.belief_propagation(graph["functions"],
                        graph["var_list"],
                        it,
                        graph["vertex"],
                        nc,ntraces)

def reset_graph_memory(graph,Nc):
    variables_list = graph["var_list"]
    for var in variables_list:
        # if node has distribution
        if "distri_orig" in var:
            # normalize
            var["distri_orig"][:,:]= (var["distri_orig"].T / np.sum(var["distri_orig"],axis=1)).T
            # clip the distribution
            np.clip(var["distri_orig"],CLIP,1,out=var["distri_orig"])

    for f in graph["functions"]:
        if f["func"] == XOR_CST:
            f["values"] = graph["publics"][f["value_label"]]
        if f["func"] == LOOKUP:
            f["table"] = graph["tables"][f["table_label"]]

if __name__ == "__main__":
    graph = create_graph("example_graph.txt")
    n = 10
   
    from tqdm import tqdm
    for nc in 2**np.arange(2,4):
        init_graph_memory(graph,n,nc)
        variables = graph["var"]
        publics = graph["publics"]
        tables = graph["tables"]
        for it in tqdm(range(1),desc="nc %d"%(nc)):
            x_0 = np.random.randint(0,nc)
            p_0 = np.random.randint(0,nc)
            x_1 = np.random.randint(0,nc)
            p_1 = np.random.randint(0,nc)
            sbox = np.random.permutation(nc).astype(np.uint32)

            k_0_expected = p_0 ^ x_0
            k_1_expected = p_1 ^ x_1
            k_2_expected = sbox[x_1] #k_1_expected ^ k_0_expected
            k_3_expected = p_0 ^ x_0
            k_4_expected = p_0 ^ x_0 ^ x_1 

            preci = (np.random.random(n)*(1 - 1/nc)).reshape(n,1) + 1/nc
            variables["p_0"]["distri_orig"][:,:] = (1-preci)/(nc-1)
            variables["p_0"]["distri_orig"][:,p_0] = preci[:,0]
            tables["sbox"][:] = sbox
            preci = (np.random.random(n)*(1 - 1/nc)).reshape(n,1) + 1/nc
            variables["x_0"]["distri_orig"][:,:] = (1-preci)/(nc-1)
            variables["x_0"]["distri_orig"][:,x_0] = preci[:,0]

            preci = (np.random.random(n)*( (1 - 1/nc) + 1/nc)).reshape(n,1)
            preci = (np.random.random(n)*(1 - 1/nc)).reshape(n,1) + 1/nc
            publics["p_1"][:] = p_1
            variables["x_1"]["distri_orig"][:,:] = (1-preci)/(nc-1)
            variables["x_1"]["distri_orig"][:,x_1] = preci[:,0]

            reset_graph_memory(graph,nc)
            
            rust.belief_propagation(graph["functions"],graph["var_list"],4,
                    graph["vertex"],
                    nc,n)

            k_0 = np.argmax(variables["k_0"]["distri"],axis=1)[0]
            k_1 = np.argmax(variables["k_1"]["distri"],axis=1)[0]
            k_2 = np.argmax(variables["k_2"]["distri"],axis=1)[0]
            k_3 = np.argmax(variables["k_3"]["distri"],axis=1)[0]
            k_4 = np.argmax(variables["k_4"]["distri"],axis=1)[0]
            assert k_0 == k_0_expected 
            assert k_1 == k_1_expected 
            assert k_2 == k_2_expected 
            #assert k_3 == k_3_expected 
            assert k_4 == k_4_expected 
