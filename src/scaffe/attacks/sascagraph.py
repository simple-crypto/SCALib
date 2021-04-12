import numpy as np
from functools import reduce
from scaffe import _scaffe_ext


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

    def __init__(self, description, n):
        self.n_ = n
        self.solved_ = False

        self.graph = SASCAGraphParser(description)
        self.nc_ = self.graph.nc
        self.tables_ = {table: None for table in self.graph.tables}
        for tab_name, init in self.graph.tables.items():
            self.set_table(tab_name, init)
        self.properties_ = self.graph.properties
        self.var_ = self.graph.var
        self.publics_ = {}

    def set_init_distribution(self, var, distribution):
        r"""Sets initial distribution of a variables.

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
            if distribution.shape != (self.n_, self.nc_):
                raise ValueError(f"Distribution for variable {var} has wrong shape")
        else:
            if distribution.shape == (self.nc_,):
                distribution = distribution.reshape((1, self.nc_))
            elif distribution.shape != (1, self.nc_):
                raise ValueError(f"Distribution for variable {var} has wrong shape")

        self.var_[var]["initial"] = distribution

    def get_distribution(self, var):
        r"""Returns the current distribution of a variable `var`. Must be
        solved beforehand with `run_bp()`.

        Parameters
        ----------
        var : string
            Identifier of the variable for which distribution must be returned.
            Distribution cannot be obtained for public variables.

        Returns
        -------
        distribution : array_like, f64
            Distribution of `var`. If `var` is SINGLE, distribution has shape
            `(1,nc)` else, has shape `(n,nc)`.
        """
        if not self.solved_:
            raise Exception("SASCAGraph not solved yet")
        return self.var_[var]["current"]

    def set_public(self, var, values):
        r"""Marks a variable `var` has public with provided `values`.

        Parameters
        ----------
        var : string
            Identifier of the variable to mark as public
        values: array_like, uint32 or int
            For MULTI variables, public values for each of the independent
            executions. Must be of shape `(n,)`.
            For SINGLE variables, public value for all executions. Must be an
            integer.
        """
        if self.var_[var]["para"]:
            if values.shape != (self.n_,):
                raise ValueError("Public data has wrong shape")
            if values.dtype != np.uint32:
                raise ValueError("Public data must be np.uint32")
        else:
            if not isinstance(values, int) and values >= 0:
                raise ValueError("SINGLE Public value must be a positive integer.")
            values = values * np.ones((self.n_,), dtype=np.uint32)
        if not np.all(values < self.nc_):
            # 0 lower bound is given by np.uint32 dtype
            raise ValueError("Values is out of [0, nc) range")
        # remove from the standard variables and move it to publics
        del self.var_[var]
        self.publics_[var] = values

    def set_table(self, table, values):
        r"""Defines a `table` content.

        Parameters
        ----------
        table : string
            Identifier of the table to fill.
        values: array_like, uint32
            Content of the table. Must be of shape `(nc,)`.
        """
        if table not in self.tables_:
            raise ValueError(f"Table {table} does not exist.")
        if values is None:
            values = None
        elif values.shape != (self.nc_,):
            raise ValueError("Table has wrong shape")
        elif values.dtype != np.uint32:
            raise ValueError("Table must be np.uint32")
        elif set(values) != set(range(self.nc_)):
            raise ValueError(
                "In current implementation, table is not a bijection over the set of values [0, nc)."
                )
        self.tables_[table] = values

    def run_bp(self, it):
        r"""Runs belief propagation algorithm on the current state of the graph.

        Parameters
        ----------
        it : int
            Number of iterations of belief propagation.
        """
        self._init_graph()
        _scaffe_ext.run_bp(
            self.properties_,
            [self.var_[x] for x in list(self.var_)],
            it,
            self.edge_,
            self.nc_,
            self.n_,
        )
        self.solved_ = True

    def _share_edge(self, property, v):
        property["neighboors"].append(self.edge_)
        self.var_[v]["neighboors"].append(self.edge_)
        self.edge_ += 1

    def _check_fully_init(self):
        for table, init in self.tables_.items():
            if init is None:
                raise ValueError(f"Table {table} not initialized")

    def _init_graph(self):
        self._check_fully_init()
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
            if property["output"] in self.publics_:
                raise ValueError(
                        "In current implementation public vars can only be ^ or & operands.\n"
                        + "Cannot assign " + property["output"]
                        )
            if len([inp for inp in property["inputs"] if inp in self.publics_]) > 1:
                raise ValueError(
                        "In current implementation there can only be one public operand."
                        )
            for inp in property["inputs"]:
                if inp in self.publics_ and property["property"] == "LOOKUP":
                    raise ValueError(
                            "In current implementation public vars can only be ^ or & operands.\n"
                            + "Cannot use " + inp + " in table lookup."
                            )
            if not any(v in self.var_ and self.var_[v]["para"] for v in property["inputs"] + [property["output"]]):
                raise ValueError(
                        "In current implementation, there must be at least one PARA var per property."
                        )


            # number of VAR MULTI in the PROPERTY
            npara = len(
                list(
                    filter(
                        lambda x: (x in self.var_) and self.var_[x]["para"],
                        [property["output"]] + property["inputs"],
                    )
                )
            )

            if property["property"] == "LOOKUP":
                property["func"] = LOOKUP
                # get the table into the function
                property["table"] = self.tables_[property["tab"]]
                # set edge to input and output
                self._share_edge(property, property["output"])
                self._share_edge(property, property["inputs"][0])

            elif property["property"] == "AND":
                # If no public inputs
                if all(x in self.var_ for x in property["inputs"]):
                    property["func"] = AND
                    self._share_edge(property, property["output"])
                    for i in property["inputs"]:
                        self._share_edge(property, i)

                # if and with public input
                elif len(property["inputs"]) == 2:
                    # AND with one public
                    property["func"] = AND_CST

                    self._share_edge(property, property["output"])

                    # which of both inputs is public
                    if property["inputs"][0] in self.var_:
                        i = (0, 1)
                    elif property["inputs"][1] in self.var_:
                        i = (1, 0)
                    else:
                        assert False

                    # share edge with non public input
                    self._share_edge(property, property["inputs"][i[0]])

                    # merge public input in the property
                    property["values"] = self.publics_[property["inputs"][i[1]]]

            elif property["property"] == "XOR":
                # if no inputs are public
                if all(x in self.var_ for x in property["inputs"]):
                    # XOR with no public
                    property["func"] = XOR

                    # share edge with the output
                    self._share_edge(property, property["output"])

                    # share edge with all the inputs
                    for i in property["inputs"]:
                        self._share_edge(property, i)

                elif len(property["inputs"]) == 2:
                    # XOR with one public
                    property["func"] = XOR_CST

                    # which of both inputs is public
                    if property["inputs"][0] in self.var_:
                        i = (0, 1)
                    elif property["inputs"][1] in self.var_:
                        i = (1, 0)
                    else:
                        assert False

                    # share edges with variables
                    self._share_edge(property, property["output"])
                    self._share_edge(property, property["inputs"][i[0]])

                    # merge public into the property
                    property["values"] = self.publics_[property["inputs"][i[1]]]

                else:
                    raise ValueError(
                            "XOR must have two operands when one operand is public."
                            )

            else:
                assert False, "Property must be either LOOKUP, AND or XOR."

        for v in self.var_:
            v = self.var_[v]
            if v["para"]:
                v["current"] = np.ones((self.n_, self.nc_))
            else:
                v["current"] = np.ones((1, self.nc_))

class SASCAGraphError(Exception):
    pass

class SASCAGraphParser():
    MAX_DISP_ERRORS=10
    def __init__(self, description):
        self.nc_decls = []
        self.var_decls = []
        self.prop_decls = []
        self.table_decls = []
        self.var = {}
        self.tables = {}
        self.properties = []
        self.errors = []

        self._parse_description(description)
        self._get_nc()
        self._build_var_set()
        self._build_tables()
        self._build_properties()

    def _build_properties(self):
        for prop_kind, res, inputs in self.prop_decls:
            prop = {
                    "property": prop_kind,
                    "output": res,
                    "neighboors": [],
                    }
            if prop_kind == "LOOKUP":
                tab = inputs[0]
                if tab not in self.tables:
                    self.errors.append(f"Table '{tab}' not declared.")
                prop["tab"] = tab
                inputs = inputs[1:]
            prop["inputs"] = inputs
            missing_vars = [v for v in inputs + [res] if v not in self.var]
            if missing_vars:
                self.errors.extend([f"Variable '{v}' not declared." for v in missing_vars])
            else:
                self.properties.append(prop)
        self._raise_errors()


    def _build_tables(self):
        for tab_name, init in self.table_decls:
            if tab_name in self.tables:
                self.errors.append(f"Table {tab_name} multiply declared.")
            elif init is None:
                self.tables[tab_name] = None
            else:
                self.tables[tab_name] = np.array(init, dtype=np.uint32)
        self._raise_errors()

    def _build_var_set(self):
        for key, para in self.var_decls:
            if key in self.var:
                self.errors.append(f"Variable {key} multiply declared.")
            else:
                self.var[key] = { "para": para == "MULTI", "neighboors": [] }
        self._raise_errors()

    def _get_nc(self):
        if len(self.nc_decls) > 1:
            self.errors.append("NC appears multiple times, can only appear once.")
        elif len(self.nc_decls) == 0:
            self.errors.append("NC not declared.")
        elif self.nc_decls[0] not in range(1, 2**16+1):
            self.errors.append("NC not in admissible range [1, 2^16].")
        else:
            self.nc = self.nc_decls[0]
        self._raise_errors()

    def _parse_description(self, description):
        errors = []
        # remove empty lines and comments
        lines = map(lambda l: l.split("#", 1)[0].strip(), description.splitlines())
        for i, line in enumerate(lines):
            if line:
                try:
                    self._parse_sasca_graph_line(line)
                except SASCAGraphError as e:
                    self.errors.append(f"Syntax Error at line {i}:'{line}'\n\t{e.args[0]}")
        self._raise_errors()

    def _parse_sasca_ident(self, ident):
        # An identifier is a non-empty string that contains only letters,
        # digits and '_', and does not start with a digit
        if not ident.isidentifier():
            raise SASCAGraphError(f"Invalid identifier '{ident}'.")
        return ident

    def _parse_sasca_int(self, x):
        try:
            return int(x)
        except ValueError:
            raise SASCAGraphError(f"Not an integer: {x}.")

    def _parse_sasca_property(self, rem):
        """Rem is all line except the initial PROPERTY."""
        # PROPERTY  res = expr
        try:
            res, prop = rem.replace(" ", "").split("=")
        except ValueError:
            raise SASCAGraphError(
                    "PROPERTY declaration should contain one '=' character."
                    )
        res = self._parse_sasca_ident(res)
        if "^" in prop:
            prop_kind = "XOR"
            inputs = prop.split("^")
        elif "&" in prop:
            prop_kind = "AND"
            inputs = prop.split("&")
            if len(inputs) != 2:
                raise SASCAGraphError(
                        "Wrong number of & operands: must be 2."
                )
        elif "[" in prop and "]" in prop:
            prop_kind = "LOOKUP"
            tab, in_ = prop.split("[")
            if not in_.endswith("]"):
                raise SASCAGraphError(
                        "Missing losing bracket of lookup expression."
                )
            inputs = [tab, in_[:-1]]
        else:
            raise SASCAGraphError("Unknown PROPERTY expression.")
        inputs = list(map(self._parse_sasca_ident, inputs))
        self.prop_decls.append((prop_kind, res, inputs))

    def _parse_sasca_graph_line(self, line):
        # Cannot fail (line is not empty).
        tag, *rem = line.replace("\t", " ").split(maxsplit=1)
        if rem:
            rem = rem[0]
        else:
            rem = ""
        if tag == "NC":
            # NC nc
            try:
                nc = int(rem.strip())
            except ValueError:
                raise SASCAGraphError("NC parameter is not an integer.")
            self.nc_decls.append(nc)
        elif tag == "VAR":
            # VAR [MULTI|SINGLE] key
            try:
                para, key = rem.split()
            except ValueError:
                raise SASCAGraphError("Wrong number of parameters to VAR declaration.")
            if para not in ("MULTI", "SINGLE"):
                raise SASCAGraphError(
                        f"Expected VAR description MULTI or SINGLE, found '{para}'."
                )
            key = self._parse_sasca_ident(key)
            self.var_decls.append((key, para))
        elif tag == "PROPERTY":
            self._parse_sasca_property(rem)
        elif tag == "TABLE":
            # TABLE name [ = "[" num, + "]"
            try:
                name, *init = rem.replace(" ", "").split("=")
            except ValueError:
                raise SASCAGraphError(
                        "Missing table name in table declaration."
                        )
            name = self._parse_sasca_ident(name)
            if init:
                try:
                    init, = init
                except ValueError:
                    raise SASCAGraphError(
                            "Multiple '=' signs in table declaration."
                            )
                if not (init.startswith("[") and init.endswith("]")):
                    raise SASCAGraphError(
                            "Table initialization not enclosed in brackets."
                            )
                init = [self._parse_sasca_int(x) for x in init[1:-1].split(",")]
            else:
                init = None
            self.table_decls.append((name, init))
        else:
            raise SASCAGraphError(f"Unknown directive '{tag}'.")

    def _raise_errors(self):
        if self.errors:
            if len(self.errors) > self.MAX_DISP_ERRORS:
                final_comment = "\n[...] {} other errors not shown.".format(
                        len(self.errors)-self.MAX_DISP_ERRORS
                        )
            else:
                final_comment = ""
            raise SASCAGraphError(
                    "\n".join(self.errors[:self.MAX_DISP_ERRORS]) +
                    final_comment
                    )

