import numpy as np
from functools import reduce
from scale import _scale_ext
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
    n : int
        The number of independent traces to process

    """
    def __init__(self,fname,nc,n):
        self.fname_ = fname
        self.nc_ = nc
        self.n_ = n

        # open file, remove empty lines and comments
        with open(fname) as fp:
            lines = map(lambda l:l.rstrip('\n').split("#",1)[0],fp.readlines())
            lines = filter(lambda l : len(l) > 0 and not l.isspace(),lines)
        
        var = {}
        tables = {}
        op = []
        for l in lines:
            tag = l.split()[0]
            if tag == "VAR":
                # VAR " " MULTI|SINGLE " " key [# comment]
                s = l.split()
                para = s[1]
                key = s[2]

                # assert para parameter is fine
                if para != "MULTI" and para != "SINGLE":
                    raise Exception("Unrecognized %s in "%(para)+l)

                var[key] = {"para":para=="MULTI","neighboors":[]}
            elif tag == "PROPERTY":
                # PROPERTY " " res " " = " "
                l = l.replace("PROPERTY","").replace(" ","")
                s = l.split("=")
                if len(s) != 2:
                    raise Exception("Not able to parse "+l)
                res = s[0]
                prop = s[1]
                
                if '^' in prop:
                    tag = "XOR"
                    if '^^' in prop:
                        raise Exception("Cannot parse line" + l)
                    inputs = prop.split('^')
                    op.append({"op":tag,"output":res,"inputs":inputs,"neighboors":[]}) 
                elif '&' in prop:
                    tag = "AND"
                    if '&&' in prop:
                        raise Exception("Cannot parse line" + l)
                    inputs = prop.split('&')
                    if len(inputs) != 2:
                        raise Exception("AND must have two operands")
                    op.append({"op":tag,"output":res,"inputs":inputs,"neighboors":[]}) 
                elif '[' in prop and ']' in prop:
                    tag = "LOOKUP"
                    tab = prop.split("[")[0]
                    inputs = [prop.split("[")[1].strip("]")]
                    op.append({"op":tag,"output":res,
                                    "inputs":inputs,"tab":tab,"neighboors":[]}) 
                else:
                    raise Exception("Not able to parse "+l)
            elif tag == "TABLE":
                # VAR " " MULTI|SINGLE " " key [# comment]
                l = l.replace("TABLE","").replace(" ","")
                if "=" in l: # intialized table
                    name = l.split("=")[0]
                    tab = l.split("=")[1]
                    if tab[0] != "[" or tab[-1] != "]":
                        raise Exception("Not able to parse table in line "+l)
                    tab = tab.replace("[","").replace("]","").split(',')
                    tab = np.array(list(map(lambda x: int(x),tab)),dtype=np.uint32)
                    if len(tab) != self.nc_:
                        raise Exception("Table must be of length nc (%d)"%(self.nc_))

                    tables[name] = tab
                else:
                    tables[l] = []
            else:
                raise Exception("Not able to parse "+l)
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
                if npara < 1:
                    raise Exception("Can have only one SINGLE per PROPERTY")
                if any(x in self.publics_ for x in op["inputs"]):
                    raise Exception("Do not support public in AND")

                op["func"] = AND
                self._share_vertex(op,op["output"])
                for i in op["inputs"]:
                    self._share_vertex(op,i)


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
