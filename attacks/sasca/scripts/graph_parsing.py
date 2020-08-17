import numpy as np
import matplotlib.pyplot as plt
from stella.attacks.sasca.Node import *
from stella.attacks.sasca.Graph import *
from tqdm import tqdm
context = []
flags =["secret","public","profile"]

def process_flag(v,flags,it=0,public=None):
    """
        NOT TO BE USED OUTSIDE THIS FILE
    """ 
    flag = flags[0].replace('#','')

    if len(flags) > 1:
        cst = int(flags[1])
    else:
        cst = 0

    if flag == "secret" or flag == "profile":
        v["node"] = VNode(0)
        v["node"]._flag["label"] = v["label"]
        v["node"]._flag["it"] = it
        if flag == "secret": v["node"]._use_log = True
    elif flag == "public":
        if public is None:
            v["node"] = np.array([cst],dtype=np.uint32)
        else:
            i = list(map(lambda p:p["label"],public)).index(v["label"])
            v["node"] = public[i]["input"][it,:]
    else:
        raise Exception("Unknown flag: ",flag, ". Should be in ", ' '.join(flags))
    v["flags"] = flag

def process_opt(v,opt,context,it=0):
    """
        NOT TO BE USED OUTSIDE THIS FILE

        Apply operations on a given line. v is the value to write,
        opt is the right og the =
        context is the list of variable within the onctext
        it is the loop iteration index
    """
    labels = list(map(lambda x:x["label"],context))
    i = labels.index(opt[0])
    v0 = context[i]["node"]
    i = labels.index(opt[2])
    v1 = context[i]["node"]

    if opt[1] == "^":
        func = bxor
    elif opt[1] == "&":
        func = band
    else:
        raise Exception("Operation not known : ",opt)

    if isinstance(v["node"],VNode) and isinstance(v1,VNode):
        apply_func(func,inputs=[v0,v1],output=v["node"])
    elif isinstance(v["node"],VNode):
        apply_func(func,inputs=[v0],offset=v1,output=v["node"])
    else:
        v["node"] = apply_func(func,inputs=[v0,v1])

def process_line(l,context,it=0,in_loop=False,public=None):
    """
        NOT TO BE USED OUTSIDE THIS FILE
        
        Parse a single line of the txt file
    """
    args = l.split()

    assert len(args)>=1
    x = args[0]

    # search is in all the variables
    # the variable to update
    labels = list(map(lambda x:x["label"],context))
    if x in labels:
        insert = False
        i = labels.index(x)
        v = context[i]
    else:
        insert = True
        v = {"label":x,
                "node":None,
                "flags":[],
                "it":it,
                "in_loop":in_loop}
        context.append(v)

    # case on the type of operation
    if args[1] == '=':
        process_opt(v,args[2:],context,it=it)
    elif args[1][0] == '#':
        process_flag(v,args[1:],it=it,public=public)
    else:
        raise Exception("Bad syntax: ",l)

def extract_flags(file_name):
    """
        Parse file_name and returns the variables labels according to the order

        public,profile,secret
    """
    with open(file_name) as fp:
        lines = list(filter(lambda l:len(l)>0,map(lambda l:l.rstrip('\n'),fp.readlines())))
    lines_loop = [None for _ in lines]

    f = False
    for i,l in enumerate(lines):
        lines_loop[i] = (l,f)
        if "#indeploop" in l:
            f = True
        elif "#endindeploop" in l:
            f = False
    
    public = list(map(lambda l: {"label":l[0].rsplit()[0],"loop":l[1]},filter(lambda l:"#public" in l[0],lines_loop)))
    profile = list(map(lambda l: {"label":l[0].rsplit()[0],"loop":l[1]},filter(lambda l:"#profile" in l[0],lines_loop)))
    secret = list(map(lambda l: {"label":l[0].rsplit()[0],"loop":l[1]},filter(lambda l:"#secret" in l[0],lines_loop)))

    return public,profile,secret

def build_graph_from_file(file_name,Nk,public=None,it=1):
    """
        Build the graph given in file_name
        Nk: field size
        public: public set of inputs
        it: number of iterations within the loop
    """
    context = []
    VNode.reset_all()
    FNode.reset_all()
    with open(file_name) as fp:
        lines = list(filter(lambda l:len(l)>0,map(lambda l:l.rstrip('\n'),fp.readlines())))

    # going through lines
    it_lines = iter(lines)
    for l in it_lines:
        if l.rsplit(' ')[0] == "#indeploop":
            loop_code = []
            for l in it_lines:
                if l != "#endindeploop":
                    loop_code.append(l)
                else:
                    break
            for it in tqdm(range(it),desc="Loop generation"):
                context_cp = [c for c in context]
                for l in loop_code:
                    process_line(l,context_cp,in_loop=True,it=it,public=public)
        else:
            process_line(l,context,public=public)
    return Graph(Nk,vnodes=VNode.buff,fnodes=FNode.buff)

def initialize_graph_from_file(graph,file_name,verbose=False,
        Nk = 256,LOOP_IT=1):

    """
        Initialize graph with description in file_name.
        return the secrets and profile distributions.

        secret: is the guessed distribution of the outputs
        profile; the distribution of the profiled variables, should be manipulated by
                the user before running BP.

    """
    public,profile,secret = extract_flags(file_name)

    if verbose:
        print("# Prepare initialized graph ...")
        print("# ... prepare profile distri")
    for v in profile: v["distri"] = np.zeros((LOOP_IT,Nk)) if v["loop"] else np.zeros((1,Nk))
    profile_lab = list(map(lambda v: v["label"],profile))
    nodes_profile = list(map(lambda v: v._flag,graph.get_nodes(lambda v: True if "label" in v._flag and v._flag["label"] in profile_lab else False)))
    nodes_profile = list(zip(nodes_profile,list(map(lambda v: profile[profile_lab.index(v["label"])]["distri"][v["it"],:],nodes_profile))))

    if verbose:
        print("# ... prepare secret distri")
    for v in secret: v["distri"] = np.zeros((LOOP_IT,Nk)) if v["loop"] else np.zeros((1,Nk))
    secret_lab = list(map(lambda v: v["label"],secret))
    nodes_secret = list(map(lambda v: v._flag,graph.get_nodes(lambda v: True if "label" in v._flag and v._flag["label"] in secret_lab else False)))
    nodes_secret = list(zip(nodes_secret,list(map(lambda v: secret[secret_lab.index(v["label"])]["distri"][v["it"],:],nodes_secret))))

    if verbose:
        print("# ... graph initializing")
    graph.initialize_nodes(nodes_profile,nodes_secret)

    if verbose:
        print("# Preparing the graph done")
    return secret,profile
