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

    Properties allows to set relationship between variables with the syntax
    `PROPERTY " " Operation`. The possible operations are listed below. We note
    the public variables cannot be assigned with PROPERTY and that a PROPERTY
    can only contain one VAR SINGLE.

    +---------+--------------+-------------------------------------------------+
    |Operation| Syntax       | Description                                     |
    +=========+==============+=================================================+
    |XOR      | `x = y^z^...`| Assigns to `x` as the XORs between the variables|
    |         |              | separated with `^`. If a public variable is used|
    |         |              | ,only two operands are allowed. Only one public |
    |         |              | operand is allowed. XOR is bitwise.             |
    +---------+--------------+-------------------------------------------------+
    |AND      | `x = y&z`    | Assigns to `x` as the AND between two variables |
    |         |              | separated with `&`. If a public variable is used|
    |         |              | only two operands are allowed. Only one public  |
    |         |              | operand is allowed. AND is bitwise.             |
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
    
    By running a belief propagation algorithm (see [1]), the distributions on all
    the variables are updated based on their initial distributions. The
    `SASCAGraph`, can be solved by using `run_bp()`.

    Notes
    -----
    [1] "Soft Analytical Side-Channel Attacks". N. Veyrat-Charvillon, B. Gérard,
    F.-X. Standaert, ASIACRYPT2014.

    Parameters
    ----------
    description : string
        String containing the graph description. 
    nc : int
        The size distributions. e.g., 256 when 8-bit variables are manipulated.
    n : int
        The number of independent traces to process within the `VAR MULTI`
        variables.

    """
    def __init__(self,description,nc,n):
        self.nc_ = nc
        self.n_ = n
        self.solved_ = False

        # remove empty lines and comments
        lines = map(lambda l:l.split("#",1)[0],description.split("\n"))
        lines = filter(lambda l : len(l) > 0 and not l.isspace(),lines)
        
        var = {}
        tables = {}
        property = []
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

                # replace spaces
                l = l.replace("PROPERTY","").replace(" ","")
                s = l.split("=")
                if len(s) != 2:
                    errors.append(("Cannot parse",l))
                    continue

                # assigned value
                res = s[0]
                # equation
                prop = s[1]
                
                if '^' in prop:
                    tag = "XOR"
                    if '^^' in prop:
                        errors.append(("Cannot parse",l))
                        continue
                    inputs = prop.split('^')
                    property.append({"property":tag,"output":res,"inputs":inputs,"neighboors":[]}) 
                elif '&' in prop:
                    tag = "AND"
                    if '&&' in prop:
                        errors.append(("Cannot parse",l))
                        continue

                    inputs = prop.split('&')
                    if len(inputs) != 2:
                        errors.append(("AND must have two operands",l))

                    property.append({"property":tag,"output":res,"inputs":inputs,"neighboors":[]}) 
                elif '[' in prop and ']' in prop:
                    tag = "LOOKUP"
                    # get table
                    tab = prop.split("[")[0]
                    
                    # get inside of the brackets
                    inputs = [prop.split("[")[1].strip("]")]
                    property.append({"property":tag,"output":res,
                                    "inputs":inputs,"tab":tab,"neighboors":[]}) 
                else:
                    errors.append(("Cannot parse",l))
                    continue

            elif tag == "TABLE":
                # VAR " " MULTI|SINGLE " " key [# comment]
                l = l.replace("TABLE","").replace(" ","")

                if "=" in l: # table has to be initialized
                    name = l.split("=")[0]
                    tab = l.split("=")[1]
                    if tab[0] != "[" or tab[-1] != "]":
                        errors.append(("Cannot parse table",l))
                        continue
                    tab = tab.replace("[","").replace("]","").split(',')
                    tab = np.array(list(map(lambda x: int(x),tab)),dtype=np.uint32)
                    if len(tab) != self.nc_:
                        errors.append(("Table must have length nc "\
                            " = %d "%(self.nc_),l),)
                        continue

                    if not np.all(tab < self.nc_):
                        errors.append(("Table contains values larger than "\
                            "%d"%(self.nc_),l))
                        continue

                    tables[name] = tab
                else:
                    # set empty table if not initialized
                    tables[l] = []
            else:
                errors.append(("Cannot parse",l))
                continue

        if len(errors) > 0:
            for err,line in errors:
                print("Error",err,":",line)
            raise Exception("Error in description parsing")

        self.tables_ = tables
        self.properties_ = property
        self.var_ = var
        self.publics_ = {}

    def set_distribution(self,var,distribution):
        r"""Sets distribution of a variables.
        
        Parameters
        ----------
        var : string
            Identifier of the variable to assign the distribution to.
        distribution : array_like, f64
            Distribution to assign. If `var` is SINGLE, must be of shape `(nc,)`
            or `(1,nc)`. If `var` is MULTI, must be of shape `(n,nc)`.
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
        r"""Returns the current distribution of a variable `var`. Must be solved
        beforehand with `run_bp()`.

        Parameters:
        -----------
        var : string
            Identifier of the variable for which distribution must be returned

        Returns:
        --------
        distribution : array_like, f64
            Distribution of `var`. If `var` is SINGLE, distribution has shape
            `(1,nc)` else, has shape `(n,nc)`. 
        """
        if not self.solved_:
            raise Exception("SASCAGraph not solved yet")
        if not var in self.var_:
            raise Exception(var+ "not found")
        return self.var_[var]["current"]

    def set_public(self,var,values):
        r"""Marks a variable `var` has public with provided `values`.

        Parameters
        ----------
        var : string
            Identifier of the variable to mark as public
        values: array_like, uint32
            Public values for each of the independent executions. Must be of
            shape `(n,)`. 
        """
        if values.shape != (self.n_,):
            raise Exception("Public data has wrong shape")
        
        if values.dtype != np.uint32:
            raise Exception("Public data must be np.uint32")
        
        # remove from the standard variables and move it to publics
        if not self.var_[var]["para"]:
            print("WARNING: VAR SINGLE to public")
        # remove public from variables
        del self.var_[var]

        self.publics_[var] = values

    def set_table(self,table,values):
        r"""Defines a `table` content.

        Parameters
        ----------
        table : string
            Identifier of the table to fill.
        values: array_like, uint32
            Content of the table. Must be of shape `(nc,)`.
        """

        if values.shape != (self.nc_,):
            raise Exception("Table has wrong shape")

        if values.dtype != np.uint32:
            raise Exception("Table must be np.uint32")
        
        if not np.all(values < self.nc_):
            raise Exception("Table contains values larger than %d"%(self.nc_))

        self.tables_[table] = values

    def run_bp(self,it):
        r"""Runs belief propagation algorithm on the current state of the graph.

        Parameters
        ----------
        it : int
            Number of iterations of belief propagation.
        """
        self._init_graph()   
        _scale_ext.run_bp(self.properties_,
                            [self.var_[x] for x in list(self.var_)],
                            it,
                            self.edge_,
                            self.nc_,self.n_)
        self.solved_ = True

    def _share_edge(self,property,v):
        property["neighboors"].append(self.edge_)
        self.var_[v]["neighboors"].append(self.edge_)
        self.edge_ += 1

    def _init_graph(self):
        # mapping to Rust functions
        AND = 0
        XOR = 1
        XOR_CST = 2
        LOOKUP = 3
        AND_CST = 4

        # edge id
        self.edge_ = 0

        # Going over all the operations. For each of the value to assign and 
        # its inputs, create a edge uniquely identified with an integer. 
        # Share the edge with the corresponding variable.
        # Check also that the conditions on the inputs are supported by the Rust
        # backend.
        for property in self.properties_:

            if property["output"] not in self.var_:
                raise Exception("Can not assign "+property["output"])

            # number of VAR MULTI in the PROPERTY
            npara = len(list(filter(lambda x: (x in self.var_)
                            and self.var_[x]["para"],
                            [property["output"]]+property["inputs"])))

            if property["property"] == "LOOKUP":
                property["func"]=LOOKUP

                # input cannot be a public
                if property["inputs"][0] not in self.var_:
                    raise Exception("Can only LOOKUP var: "+property["inputs"][0])
                # if npara == 0 means that both input and output are SINGLE
                if npara == 0:
                    raise Exception("Can have only one SINGLE per PROPERTY")
                if len(self.tables_[property["tab"]]) != self.nc_:
                    raise Exception("Table "+property["tab"]+" used but unset")

                # get the table into the function
                property["table"] = self.tables_[property["tab"]]
    
                # set edge to input and output
                self._share_edge(property,property["output"])
                self._share_edge(property,property["inputs"][0])

            elif property["property"] == "AND":
                if len(property["inputs"]) != 2:
                    raise Exception("AND can only have 2 inputs")

                # If no public inputs
                if all(x in self.var_ for x in property["inputs"]):
                    if npara < len(property["inputs"]):
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # no public AND
                    property["func"] = AND
                    self._share_edge(property,property["output"])
                    for i in property["inputs"]:
                        self._share_edge(property,i)

                # if and with public input 
                elif len(property["inputs"]) == 2:
                    # if npara == 0 means that both input and output are SINGLE
                    if npara == 0:
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # XOR with one public
                    property["func"] = AND_CST

                    self._share_edge(property,property["output"])
                    
                    # which of both inputs is public
                    if property["inputs"][0] in self.var_:
                        i = (0,1)
                    elif property["inputs"][1] in self.var_: 
                        i = (1,0)
                    else:
                        raise Exception("Not able to process",property)

                    # share edge with non public input 
                    self._share_edge(property,property["inputs"][i[0]])

                    # merge public input in the property
                    if len(self.publics_[property["inputs"][i[1]]]) != self.n_:
                        raise Exception("Public %s with" \
                                "length different than n"%(property["inputs"][i[1]]))
                    property["values"] = self.publics_[property["inputs"][i[1]]]


            elif property["property"] == "XOR":
                # if no inputs are public
                if all(x in self.var_ for x in property["inputs"]):
                    # XOR with no public
                    property["func"] = XOR
                    
                    if npara < len(property["inputs"]):
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # share edge with the output 
                    self._share_edge(property,property["output"])

                    # share edge with all the inputs
                    for i in property["inputs"]:
                        self._share_edge(property,i)
                
                elif len(property["inputs"]) == 2:
                    # XOR with one public
                    property["func"] = XOR_CST

                    if npara == 0:
                        raise Exception("Can have only one SINGLE per PROPERTY")

                    # which of both inputs is public
                    if property["inputs"][0] in self.var_:
                        i = (0,1)
                    elif property["inputs"][1] in self.var_: 
                        i = (1,0)
                    else:
                        raise Exception("Not able to process",property)

                    # share edges with variables
                    self._share_edge(property,property["output"])
                    self._share_edge(property,property["inputs"][i[0]])

                    # merge public into the property
                    if len(self.publics_[property["inputs"][i[1]]]) != self.n_:
                        raise Exception("Public %s with" \
                                "length different than n"%(property["inputs"][i[1]]))
                    property["values"] = self.publics_[property["inputs"][i[1]]]
                else:
                    raise Exception("Not able to process",property)

        for v in self.var_:
            v = self.var_[v]
            if v["para"]:
                v["current"] = np.ones((self.n_,self.nc_))
            else:
                v["current"] = np.ones((1,self.nc_))
 
