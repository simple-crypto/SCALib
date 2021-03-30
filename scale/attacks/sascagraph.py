import numpy as np
from functools import reduce
from scale import _scale_ext
class SASCAGraph:
    r"""SASCAGraph allows to run Soft Analytical Side-Channel Attacks (SASCA).
    It takes as input a string that represents the implementation (or graph) to
    evaluate. Namely, it contains the intermediate variables (VAR) within the
    implementations and explicits the operations (PROPERTY) that link them. Once
    the description is loaded, and so the SASCAGraph created, user can mark
    variables as PUBLIC (e.g., plaintext), set their distribution (e.g., Sbox
    output) and specify table lookups (e.g. Sbox of the cipher).
    Through the flags SINGLE and MULTI, SASCAGraph description allows to mark
    multiple independent execution of described circuit.

    The variables within the computation are represented with VAR with the
    syntax `VAR " " MULTI|SINGLE " " name`. We note that public variables can
    only the MULTI. We describe the attributes below:

    +------------+-------------------------------------------------------------+
    | Key word   | Meaning                                                     |
    +============+=============================================================+
    | SINGLE     | This variable is unique for all the executions of the       |
    |            | circuit. Encryption keys are usually marked as SINGLE.      |
    +------------+-------------------------------------------------------------+
    | MULTI      | This variable changes at every execution of the circuit.    |
    |            | Intermediate variables of the encryption (e.g., Sbox output)|
    |            | are usually marked as MULTI.                                |
    +------------+-------------------------------------------------------------+
    | name       | Unique identifier for the intermediate variable. It must    |
    |            | be unique.                                                  |
    +------------+-------------------------------------------------------------+

    Tables can also be included in the graph description with the syntax 
    `TABLE name ["=[,,,]"]` that we describe below:

    +------------+-------------------------------------------------------------+
    | Key word   | Meaning                                                     |
    +============+=============================================================+
    | name       | Unique identifier for the intermediate variable. It must    |
    |            | be unique.                                                  |
    +------------+-------------------------------------------------------------+
    | =[,...,]   | Optional description of the table. The table is described   |
    |            | with comma separated integers. The size of the table must be|
    |            | equal to `nc` and all its contains smalled than `nc`. Table |
    |            | can also be specified through `set_table()`.                |
    +------------+-------------------------------------------------------------+

    Properties linking the different variables are listed bellow with their
    corresponding syntax. We note the public variables cannot be assigned with
    PROPERTY and that a PROPERTY can only contain one VAR SINGLE.

    +---------+--------------+-------------------------------------------------+
    |Operation| Syntax       | Description                                     |
    +=========+==============+=================================================+
    |XOR      | `x = y^z^...`| Assigns to `x` as the XORs between the variables|
    |         |              | separated with `^`. If a public variable is used|
    |         |              | only two operands are allowed. Only one public  |
    |         |              | is allowed. XOR is bitwise.                     |
    +---------+--------------+-------------------------------------------------+
    |AND      | `x = y&z`    | Assigns to `x` as the AND between two variables |
    |         |              | separated with `&`. If a public variable is used|
    |         |              | only two operands are allowed. Only one public  |
    |         |              | is allowed. AND is bitwise.                     |
    +---------+--------------+-------------------------------------------------+
    |LOOKUP   | `x = t[y]`   | Assigns to `x` the value at index `y` in table  |
    |         |              | `t`. No public variable is allowed in this      |
    |         |              | PROPERTY.                                       |
    +---------+--------------+-------------------------------------------------+


    An attack attempting to recover the secret key byte `k` can be expressed
    with the following pseudo code. `sbox` is the Sbox of the blockcipher, `p` the
    plaintext, `x` the Sbox input and `y` the Sbox output.

    .. code-block:: 

        # Describe and generate the SASCAGraph
        graph_desc = ´´´ 
            # Small unprotected Sbox example
            TABLE sbox   # The Sbox
            VAR SINGLE k # The key
            VAR MULTI p  # The plaintext
            VAR MULTI x 
            VAR MULTI y
            PROPERTY x = sbox[y] # Sbox lookup
            PROPERTY x = k ^ p   # Key addition
            ´´´
        graph = SASCAGraph(graph_desc,256,n)
        
        # Encode data into the graph
        graph.set_table("sbox",aes_sbox)
        graph.set_public("p",plaintexts)
        graph.set_distribution("y",x_distribution)
        
        # Solve graph
        graph.run_bp(it=3)

        # Get key distribution and derive key guess
        k_distri = graph.get_distribution("k")
        key_guess = np.argmax(k_distri[0,:])
    
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
        String containing the graph description. 
    nc : int
        The size distributions. e.g., 256 when 8-bit variables are manipulated.
    n : int
        The number of independent traces to process within the `VAR MULTI`
        variables.

    """
    def __init__(self,string,nc,n):
        self.nc_ = nc
        self.n_ = n

        # remove empty lines and comments
        lines = map(lambda l:l.split("#",1)[0],string.split("\n"))
        lines = filter(lambda l : len(l) > 0 and not l.isspace(),lines)
        
        var = {}
        tables = {}
        op = []
        errors = []
        for l in lines:
            tag = l.split()[0]
            if tag == "VAR":
                # VAR " " MULTI|SINGLE " " key
                s = l.split()
                para = s[1]
                key = s[2]
                if key in var:
                    errors.append(("Var "+key+" already defined",l))
                    continue

                # assert para parameter is fine
                if para != "MULTI" and para != "SINGLE":
                    errors.append(("Unrecognized "+para, l))
                    continue

                var[key] = {"para":para=="MULTI","neighboors":[]}
            elif tag == "PROPERTY":
                # PROPERTY " " res " " = " "
                l = l.replace("PROPERTY","").replace(" ","")
                s = l.split("=")
                if len(s) != 2:
                    errors.append(("Cannot parse",l))
                    continue

                res = s[0]
                prop = s[1]
                
                if '^' in prop:
                    tag = "XOR"
                    if '^^' in prop:
                        errors.append(("Cannot parse",l))
                        continue
                    inputs = prop.split('^')
                    op.append({"op":tag,"output":res,"inputs":inputs,"neighboors":[]}) 
                elif '&' in prop:
                    tag = "AND"
                    if '&&' in prop:
                        errors.append(("Cannot parse",l))
                        continue

                    inputs = prop.split('&')
                    if len(inputs) != 2:
                        errors.append(("AND must have two operands",l))

                    op.append({"op":tag,"output":res,"inputs":inputs,"neighboors":[]}) 
                elif '[' in prop and ']' in prop:
                    tag = "LOOKUP"
                    tab = prop.split("[")[0]
                    inputs = [prop.split("[")[1].strip("]")]
                    op.append({"op":tag,"output":res,
                                    "inputs":inputs,"tab":tab,"neighboors":[]}) 
                else:
                    errors.append(("Cannot parse",l))
                    continue

            elif tag == "TABLE":
                # VAR " " MULTI|SINGLE " " key [# comment]
                l = l.replace("TABLE","").replace(" ","")
                if "=" in l: # intialized table
                    name = l.split("=")[0]
                    tab = l.split("=")[1]
                    if tab[0] != "[" or tab[-1] != "]":
                        errors.append(("Cannot parse table",l))
                        continue
                    tab = tab.replace("[","").replace("]","").split(',')
                    tab = np.array(list(map(lambda x: int(x),tab)),dtype=np.uint32)
                    if len(tab) != self.nc_:
                        errors.append(("Table must have length nc"\
                            "(%d)"%(self.nc_),l),)
                        continue

                    if not np.all(tab < self.nc_):
                        errors.append(("Table contains values larger than "\
                            "%d"%(self.nc_),l))
                        continue

                    tables[name] = tab
                else:
                    tables[l] = []
            else:
                errors.append(("Cannot parse",l))
                continue

        if len(errors) > 0:
            for err,line in errors:
                print("Error",err,":",line)
            raise Exception("Error in description parsing")

        self.tables_ = tables
        self.op_ = op
        self.var_ = var
        self.publics_ = {}

    def set_distribution(self,var,distribution):
        r"""Sets distribution of a variables. To modify a distribution, the
        returned array must be modified. 

        Parameters
        ----------
        var : string
            Label of an variable with a distribution (nor public nor table).
        distribution : array_like, f64
            TODO
        """
        para = self.var_[var]["para"]
        if para:
            shape_exp = (self.n_,self.nc_)
            if distribution.shape != shape_exp:
                raise Exception("Distribution has wrong shape")
        elif distribution.shape == (self.nc_,):
            distribution = distribution.reshape((1,self.nc_))
        elif distribution.shape != (1,self.nc_):
            raise Exception("Distribution has wrong shape")

        self.var_[var]["initial"] = distribution
    
    def get_distribution(self,var):
        if not var in self.var_:
            raise Exception(var+ "not found")
        return self.var_[var]["current"]

    def set_public(self,public,values):
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
        if values.shape != (self.n_,):
            raise Exception("Public data has wrong shape")
        
        if values.dtype != np.uint32:
            raise Exception("Public data must be np.uint32")

        if not self.var_[public]["para"]:
            print("WARNING: VAR SINGLE to public")
        # remove public from variables
        del self.var_[public]

        self.publics_[public] = values

    def set_table(self,table,values):
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

        if values.shape != (self.nc_,):
            raise Exception("Table has wrong shape")

        if values.dtype != np.uint32:
            raise Exception("Table must be np.uint32")
        
        if not np.all(values < self.nc_):
            raise Exception("Table contains values larger than %d"%(self.nc_))

        self.tables_[table] = values

    def _share_vertex(self,op,v):
        op["neighboors"].append(self.vertex_)
        self.var_[v]["neighboors"].append(self.vertex_)
        self.vertex_ += 1

    def _init_graph(self):
        # mapping to Rust functions
        AND = 0
        XOR = 1
        XOR_CST = 2
        AND_CST = 4
        LOOKUP = 3

        # vertex id
        self.vertex_ = 0
        for op in self.op_:
            if op["output"] not in self.var_:
                raise Exception("Can not assign "+op["output"])
            npara = len(list(filter(lambda x: (x in self.var_)
                            and self.var_[x]["para"],
                            [op["output"]]+op["inputs"]))) 
            if op["op"] == "LOOKUP":
                op["func"]=LOOKUP
                if op["inputs"][0] not in self.var_:
                    raise Exception("Can only LOOKUP var: "+op["inputs"][0])
                if npara == 0:
                    raise Exception("Can have only one SINGLE per PROPERTY")
                if len(self.tables_[op["tab"]]) != self.nc_:
                    raise Exception("Table "+op["tab"]+" used but unset")

                # get the table into the function
                op["table"] = self.tables_[op["tab"]]
    
                # set vertex to input and output
                self._share_vertex(op,op["output"])
                self._share_vertex(op,op["inputs"][0])

            elif op["op"] == "AND":
                # AND
                if len(op["inputs"]) != 2:
                    raise Exception("AND can only have 2 inputs")

                if all(x in self.var_ for x in op["inputs"]):
                    if npara < len(op["inputs"]):
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # no public AND
                    op["func"] = AND
                    self._share_vertex(op,op["output"])
                    for i in op["inputs"]:
                        self._share_vertex(op,i)
                
                elif len(op["inputs"]) == 2:
                    if npara == 0:
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # XOR with one public
                    op["func"] = AND_CST
                    self._share_vertex(op,op["output"])
                    if op["inputs"][0] in self.var_:
                        i = (0,1)
                    elif op["inputs"][1] in self.var_: 
                        i = (1,0)
                    else:
                        raise Exception("Not able to process",op)
                    self._share_vertex(op,op["inputs"][i[0]])

                    if len(self.publics_[op["inputs"][i[1]]]) != self.n_:
                        raise Exception("Public %s with" \
                                "length different than n"%(op["inputs"][i[1]]))
                    op["values"] = self.publics_[op["inputs"][i[1]]]


            elif op["op"] == "XOR":
                if all(x in self.var_ for x in op["inputs"]):
                    if npara < len(op["inputs"]):
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # no public XOR
                    op["func"] = XOR
                    self._share_vertex(op,op["output"])
                    for i in op["inputs"]:
                        self._share_vertex(op,i)
                
                elif len(op["inputs"]) == 2:
                    if npara == 0:
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # XOR with one public
                    op["func"] = XOR_CST
                    self._share_vertex(op,op["output"])
                    if op["inputs"][0] in self.var_:
                        i = (0,1)
                    elif op["inputs"][1] in self.var_: 
                        i = (1,0)
                    else:
                        raise Exception("Not able to process",op)
                    self._share_vertex(op,op["inputs"][i[0]])

                    if len(self.publics_[op["inputs"][i[1]]]) != self.n_:
                        raise Exception("Public %s with" \
                                "length different than n"%(op["inputs"][i[1]]))
                    op["values"] = self.publics_[op["inputs"][i[1]]]
                else:
                    raise Exception("Not able to process",op)

        for v in self.var_:
            v = self.var_[v]
            if v["para"]:
                v["current"] = np.ones((self.n_,self.nc_))
            else:
                v["current"] = np.ones((1,self.nc_))
 
    def run_bp(self,it):
        r"""Runs belief propagation algorithm on the current state of the graph.
        Updates the `current` distribution for all the variables. Note that only
        the `initial` distributions are taken as input for belief propagation.

        Parameters
        ----------
        it : int
            Number of iterations of belief propagation.
        """
        self._init_graph()   
        _scale_ext.run_bp(self.op_,
                            [self.var_[x] for x in list(self.var_)],
                            it,
                            self.vertex_,
                            self.nc_,self.n_)
