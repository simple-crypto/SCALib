import numpy as np
import scale.lib.scale as rust
class SASCAGraph:
    r"""SASCAGraph allows to run Soft Analytical Side-Channel Attacks (SASCA).
    It takes as input a .txt file that represent the implementation (or graph)
    to evaluate.  Namely, it contains the intermediate variables within the
    implementations and explicits the operations that links them.

    In the description files, the variable `x` can be tagged with various flags
    such that `x [#TAG]`. If a variable has multiple tags, they must be declared
    on multiple lines.  Variables must be uniquely identified since `SASCAGraph`
    does not support shadowing.

    +------------+----------------------------------------------+--------------+
    | Tag        | Meaning                                      | Has distri.  |
    +============+==============================================+==============+
    |`#secret`   | Secret variable (e.g. key). After the attack,|    Yes       |  
    |            | the secret distribution is stored in the     |              |
    |            | `current` distribution.                      |              |
    +------------+----------------------------------------------+--------------+
    |`#profile`  | Variable that is profiled by the adversary.  |    Yes       |
    |            | Initial variable distribution must then be   |              |
    |            | set in `initial` distribution which has a    |              |
    |            | shape `(n,nc)`. This can be set to the output|              |
    |            | of a `predict_proba()` call.                 |              |
    +------------+----------------------------------------------+--------------+
    |`#public`   | The variable is a public input               |     No       |
    |            | (e.g., plaintext). Must be set to an array   |              |
    |            | of shape `(n,)`.                             |              |
    +------------+----------------------------------------------+--------------+
    |`#table`    | Represents a public table (e.g., Sbox). Must |     No       |
    |            | be set to an array of shape `(nc,)`.         |              |
    +------------+----------------------------------------------+--------------+
    |  /         | A variable can also be implicitly declared as|    Yes       | 
    |            | the output of a function (see next table).   |              |
    |            | If no flag specified, then it also has a     |              |
    |            | `current` distribution.                      |              |
    +------------+----------------------------------------------+--------------+
    
    Multiple operations can be performed on variables, they are described in the
    following table.

    +------------+-------------+-----------------------------------------------+
    |Operation   | Syntax      | Description                                   |
    +============+=============+===============================================+
    |Bitwise XOR | `x = y ^ z` | `x,y,z` must have a distribution.             |
    |            |             | Describes bitwise XOR between all variables.  |
    |            |             | Can represent XOR between arbitrary number of |
    |            |             | variables in a single line.                   |
    +------------+-------------+-----------------------------------------------+ 
    |Bitwise AND | `x = y & z` | `x,y,z` must have distributions.              |
    |            |             | Describes bitwise AND.                        |
    +------------+-------------+-----------------------------------------------+
    |Table lookup| `x = y -> t`| `x` and `y` must have a distribution.         |
    |            |             | `t` must be a table. Represents the           |
    |            |             | table lookup such that `x=t[y]`.              |
    +------------+-------------+-----------------------------------------------+
    |Public XOR  | `x = y + p` | `x` and `y` must have a distribution.         |
    |            |             | `p` must be `#public`. Performs XOR           |
    |            |             | between the `p` and `y`.                      |
    +------------+-------------+-----------------------------------------------+
   
    The flag `#indeploop` means that the following block is repeated `n` times.
    This block must the ended with `#endindeploop`. Tables must be declared
    outside of the loop. Publics must be declared inside of the loops. Profile
    must be declared inside of the loop. Secret can be declared either outside
    or inside of the loop. In the second case, its `current` distribution is of
    shape `(1,nc)`.
    
    An attack attempting to recover the secret key byte `k` can be expressed
    with the following code. `sbox` is the Sbox of the blockcipher, `p` the
    plaintext, `x` the Sbox input and `y` the Sbox output.

    .. code-block::
        
        k #secret
        sbox #table

        #indeploop

        p #public
        y #profile
        x = k + p
        y = x -> sbox

        #endindeploop
    
    By running a belief propagation algorithm (see [1]), the distribution on all
    the variables are updated based on their initial distributions. For
    `SASCAGraph`, this is done with `run_bp()`.

    Notes
    -----
    [1] "Soft Analytical Side-Channel Attacks". N. Veyrat-Charvillon, B. Gérard,
    F.-X. Standaert, ASIACRYPT2014.

    Parameters
    ----------
    fname : string
        The file that contains the graph description.
    nc : int
        The size distributions. e.g., 256 when 8-bit variables are manipulated.

    """
    def __init__(self,fname,nc):
        self.fname_ = fname
        self.nc_ = nc

        self.graph_ = _create_graph(fname)
        self.n_ = 0
        self.initialized_ = False

    def init_graph_memory(self,n):
        r"""Initialize the internal arrays for tables, publics and variables.
        They then can be accessed / modified through `get_variable()`,
        `get_public()` and `get_table()`.

        Parameters
        ----------
        n : int
            Number of iterations in the #indeploop.
        """
        self.n_ = n
        self.initialized_ = True
        _init_graph_memory(self.graph_,n,self.nc_)

    def get_distribution(self,var,key):
        r"""Returns distribution of a variables. To modify a distribution, the
        returned array must be modified. 

        Parameters
        ----------
        var : string
            Label of an variable with a distribution (nor public nor table).
        key : string
            Internal array to return. `current` for the current estimated
            distribution. `initial` for the initial distribution of profiled
            variable.

        Returns
        -------
        distribution : array_like, f64
            Requested distribution of `var`. Has shape `(n,nc)` if variable
            declared in the loop, `(1,nc)` otherwise.
        """
        
        if not self.initialized_:
            raise Exception("SASCAGraph not initialized")

        return self.graph_["var"][var][key]

    def get_public(self,p):
        r"""Returns the array representing public data. To modify public data,
        the returned array must be modified.

        Parameters
        ----------
        p : string
            Label of public variable to return.

        Returns
        -------
        data : array_like, uint32
            Internal array for the public data `p`. Array is of shape `(n,)`.
        """
        
        if not self.initialized_:
            raise Exception("SASCAGraph not initialized")

        return self.graph_["publics"][p]

    def get_table(self,t):
        r"""Returns the array representing a table lookup. To modify a table, 
        the returned array must be modified.

        Parameters
        ----------
        p : string
            Label of the table to return.

        Returns
        -------
        data : array_like, uint32
            Internal array for the table `t`. Array is of shape `(nc,)`.
        """

        if not self.initialized_:
            raise Exception("SASCAGraph not initialized")

        return self.graph_["tables"][t]
    
    def get_secret_labels(self):
        r"""Return a label for all the secret variables.

        Returns
        -------
        labels : array_like, string
            All the labels that are flagged as `#secret`
        """
        var = self.graph_["var"]
        return list(filter(lambda x: var[x]["flags"] & SECRET != 0,var))

    def get_profile_labels(self):
        r"""Return a label for all the profile variables.

        Returns
        -------
        labels : array_like, string
            All the labels that are flagged as `#profile`
        """
        var = self.graph_["var"]
        return list(filter(lambda x: var[x]["flags"] & PROFILE != 0,var))

    def get_public_labels(self):
        r"""Return a label for all the public variables.

        Returns
        -------
        labels : array_like, string
            All the labels that are flagged as `#public`
        """
        var = self.graph_["var"]
        return list(self.graph_["publics"])
        

    def run_bp(self,it):
        r"""Runs belief propagation algorithm on the current state of the graph.
        Updates the `current` distribution for all the variables. Note that only
        the `initial` distributions are taken as input for belief propagation.

        Parameters
        ----------
        it : int
            Number of iterations of belief propagation.
        """
        if not self.initialized_:
            raise Exception("SASCAGraph not initialized")

        graph = self.graph_
        _reset_graph_memory(graph,self.nc_)
        rust.run_bp(graph["functions"],
                            graph["var_list"],
                            it,
                            graph["vertex"],
                            self.nc_,self.n_)

###########################
# PRIVATE Helper Methods 
###########################
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

SECRET = 1 << 0
PUBLIC = 1 << 1
PROFILE = 1 << 2
TABLE = 1 << 3

CLIP = 1E-50

def _new_variable(i):
    return {"id":i,"neighboors":[],"flags":0,"in_loop":False}
def _new_function(i,func):
    return {"id":i,"inputs":[],"outputs":[],"func":func}

def _init_graph_memory(graph,N,Nc):
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
                if "initial" in var: del var["initial"]
                var["initial"] = np.ones((n,Nc))
            if "current" in var: del var["current"]
            var["current"] = np.zeros((n,Nc))

    for p in graph["publics"]:
        graph["publics"][p] = np.zeros(N,dtype=np.uint32)
    for p in graph["tables"]:
        graph["tables"][p] = np.zeros(Nc,dtype=np.uint32)
 
def _create_graph(fname):
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
            node = _new_variable(len(variables))
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
            func = _new_function(i,f["val"])
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

def _reset_graph_memory(graph,Nc):
    variables_list = graph["var_list"]
    for var in variables_list:
        # if node has distribution
        if "initial" in var:
            # normalize
            var["initial"][:,:]= (var["initial"].T / np.sum(var["initial"],axis=1)).T
            # clip the distribution
            np.clip(var["initial"],CLIP,1,out=var["initial"])

    for f in graph["functions"]:
        if f["func"] == XOR_CST:
            f["values"] = graph["publics"][f["value_label"]]
        if f["func"] == LOOKUP:
            f["table"] = graph["tables"][f["table_label"]]

