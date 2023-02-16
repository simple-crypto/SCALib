from typing import Sequence, Mapping, Union, Optional

import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext

__all__ = ["FactorGraph", "BPState"]

CstValue = Union[int, Sequence[int]]
ValsAssign = Mapping[str, CstValue]


class FactorGraph:
    r"""FactorGraph allows to run Soft Analytical Side-Channel Attacks (SASCA).

    A SASCA is based on a set of variables, on knowledge of
    relationships between those variables and information about the values of
    the variables.

    Variables have values in a binary finite field of size `nc`, and are viewed
    as bit vectors. Those values are represented as integers in [0, nc).
    Variables can be qualified as `SINGLE` or `MULTI`, which relates
    to the multiple-execution feature of `FactorGraph`: when performing a
    SASCA, it is useful to acquire multiple execution traces. In these
    executions, some variables stay the same (e.g. an encryption key), and other
    change (e.g. plaintext, masked variables, etc.).
    The `SINGLE` qualifier should be used for variables that remain the same
    and `MULTI` for variables that change.
    `FactorGraph` will then build a graph where each `MULTI` variable is
    replicated `n` times (once for each execution), as well as all the
    relationships that relate at least on `MULTI` variable.

    Relationships between variables are bitwise XOR, bitwise AND, bitwise OR,
    bitiwise negation, modular addition, modular multiplication and lookup
    table. A lookup table can describe any function that maps a single variable
    to another variable.
    Description of `nc`, the variables, and the relationships is given in a
    text format specified below.

    Finally, variables are of two kinds: the public values and the variables.
    The public values have a single known value, while variables have an
    uncertain value modelled as a probability distribution.
    The SASCA computes these distributions from the relationships encoded by
    the graph, and from prior information.
    The prior distributions are by default uniform, but this can be changed.

    An attack attempting to recover the secret key byte `k` is shown below.

    >>> # Describe and generate the SASCAGraph
    >>> graph_desc = '''
    ...     # Small unprotected Sbox example
    ...     NC 256 # Graph over GF(256)
    ...     TABLE sbox   # Sbox
    ...     VAR SINGLE k # key (to recover !)
    ...     PUB MULTI p  # plaintext (known)
    ...     VAR MULTI x # Sbox input
    ...     VAR MULTI y # Sbox output (whose leakage is targeted)
    ...     PROPERTY x = k ^ p   # Key addition
    ...     PROPERTY y = sbox[x] # Sbox lookup
    ...     '''
    >>> sbox = np.arange(256, dtype=np.uint32) # don't use this S-box ;)
    >>> graph = FactorGraph(graph_desc, {"sbox": sbox})
    >>> # n is the number of traces for our attack.
    >>> n = 2
    >>> plaintexts = np.array([0, 1], dtype=np.uint32)
    >>> bp = BPState(graph, n, {"p": plaintexts})
    >>> y_leakage = np.random.rand(2, 256) # this might come from an LDA
    >>> y_leakage = y_leakage / y_leakage.sum(axis=1, keepdims=True)
    >>> bp.set_evidence("y",y_leakage)
    >>> # Solve graph
    >>> bp.bp_loopy(it=3, initialize_states=True)
    >>> # Get key distribution and derive key guess
    >>> k_distri = bp.get_distribution("k")
    >>> key_guess = np.argmax(k_distri)

    By running a belief propagation algorithm (see [1]_), the distributions on all
    the variables are updated based on their initial distributions. The
    `SASCAGraph` can be solved by using `run_bp()`.

    Notes
    -----

    **The graph description format** is a text line-oriented format (each line
    of text is a statement) that describes a set of variables and relationships
    between those variables.
    The ordering of the statements is irrelevant and whitespace is irrelevant
    except for newlines and around keywords. End-of-line comments start with
    the `#` symbol.
    The statements are:

    - `NC <nc>`: specifies the field size (must be a power of two). There must
      be one `NC` statement in the description.
    - `PUB SINGLE|MULTI variable_name`: declares a public variable.  `variable_name`
      is an identifier of the variable (allowed characters are letters, digits
      and underscore). One of the qualifiers `SINGLE` or `MULTI` must be given.
    - `VAR SINGLE|MULTI variable_name`: declares a variables.
    - `PROPERTY w = x^y^z`: declares a bitwise XOR property. There can be any
      number of operands.
    - `PROPERTY x = x&y`: declares a bitwise AND property.
    - `PROPERTY x = t[y]`: declares a LOOKUP property (`y` is the lookup of the
      table `t` at index `y`). No public variable is allowed in this property.
    - `PROPERTY x = !y`: declares a bitwise NOT property.
      No public variable is allowed in this property.
    - `TABLE` t = [0, 3, 2, 1]`: Declares a table that can be used in a LOOKUP.
      The values provided in the table must belong to the interval [0, nc).
      The initialization expression can be omitted from the graph description
      (e.g. `TABLE t`) and be given with `tables` parameter.


    **Note**: if the `MULTI` feature doesn't match your use-case, using only
    `SINGLE` variables works (or set the number of execution to 1).


    .. [1] "Soft Analytical Side-Channel Attacks". N. Veyrat-Charvillon, B.
       Gérard, F.-X. Standaert, ASIACRYPT2014.

    Parameters
    ----------
    graph: string
        The graph description.
    """

    def __init__(
        self,
        graph_text: str,
        tables: Optional[Mapping[str, npt.NDArray[np.uint32]]] = None,
    ):
        if tables is None:
            tables = dict()
        self._inner = _scalib_ext.FactorGraph(graph_text, tables)

    def sanity_check(self, pub_assignment: ValsAssign, var_assignment: ValsAssign):
        """Verify that the graph is compatible with example variable assignments.

        If the graph is not compatible, raise a ``ValueError``.

        Parameters
        ----------
        pub_assignment:
            For each public variable its value for all test executions.
        var_assignment:
            For each non-public variable its value for all test executions.

        Returns
        -------
        None
        """
        self._inner.sanity_check(pub_assignment, var_assignment)


class BPState:
    """Belief propagation state associated to a :class:`FactorGraph`.

    This is a stateful object on which belief propagation operations can be run.
    See :class:`scalib.attacks.FactorGraph` for usage example.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        nexec: int,
        public_values: Optional[ValsAssign] = None,
    ):
        if public_values is None:
            public_values = dict()
        self._inner = factor_graph._inner.new_bp(nexec, public_values)

    def set_evidence(self, var: str, distribution: Optional[npt.NDArray[np.float64]]):
        r"""Sets prior distribution of a variable.

        Parameters
        ----------
        var :
            Identifier of the variable to assign the distribution to.
        distribution :
            Distribution to assign. If `var` is SINGLE, must be of shape `(nc,)`.
            If `var` is MULTI, must be of shape `(nexec,nc)`.
            If None, sets the prior distribution to uniform.
        """
        if distribution is None:
            self._inner.drop_evidence(var)
        else:
            self._inner.set_evidence(var, distribution)

    def bp_loopy(self, it: int, initialize_states: bool):
        """Runs belief propagation algorithm on the current state of the graph.

        Parameters
        ----------
        it :
            Number of iterations of belief propagation.
        initialize_states:
            Whether to update variable distributions before running the BP iterations.
            Recommended after using :func:`BPState.set_evidence`.
        """
        if initialize_states:
            self._inner.propagate_all_vars()
        self._inner.propagate_loopy_step(it)

    def bp_acyclic(
        self,
        dest: str,
        *,
        clear_intermediates: bool = True,
        clear_evidence: bool = False,
    ):
        """Runs the non-loopy belief propagation algorithm on the current state of the graph.
        This only works if the graph is not cyclic.

        Parameters
        ----------
        dest:
            Variable for which the belief propagation is computed.
        clear_intermediates:
            Drop the intermetidate distributions and beliefs that are computed.
        clear_evidence:
            Drop the evidence for the variables, once used in the algorithm.
        """
        self._inner.propagate_acyclic(dest, clear_intermediates, clear_evidence)

    def get_distribution(self, var: str) -> Optional[npt.NDArray[np.float64]]:
        r"""Returns the current distribution of a variable `var`.

        Parameters
        ----------
        var : string
            Identifier of the variable for which distribution must be returned.
            Distribution cannot be obtained for public variables.

        Returns
        -------
        distribution : array_like, f64
            Distribution of `var`. If `var` is SINGLE, distribution has shape
            `(nc)`. Else, it has shape `(n,nc)`.
            If the variable has a uniform distribution, None may be returned
            (but this is not guaranteed).
        """
        return self._inner.get_state(var)

    def is_cyclic(self) -> bool:
        """Test is the graph is cyclic."""
        return self._inner.is_cyclic()

    def set_distribution(
        self, var: str, distribution: Optional[npt.NDArray[np.float64]]
    ):
        r"""Sets current distribution of a variable in the BP.

        Parameters
        ----------
        var :
            Identifier of the variable to assign the distribution to.
        distribution :
            Distribution to assign. If `var` is SINGLE, must be of shape `(nc,)`.
            If `var` is MULTI, must be of shape `(nexec,nc)`.
            If None, sets the distribution to uniform.
        """
        if distribution is None:
            self._inner.drop_state(var)
        else:
            self._inner.set_state(var, distribution)

    def get_belief_to_var(
        self, var: str, factor: str
    ) -> Optional[npt.NDArray[np.float64]]:
        r"""Returns the current belief from factor to var.

        Parameters
        ----------
        var : string
            Identifier of the variable for which distribution must be returned.
        factor : string
            Identifier of the factor for which distribution must be returned.

        Returns
        -------
        distribution : array_like, f64
            Belief on the edge from `factor` to `var`. If `factor` is SINGLE, distribution has shape
            `(nc)`. Else, it has shape `(n,nc)`.
            If the belief is a uniform distribution, None may be returned
            (but this is not guaranteed).
        """
        return self._inner.get_belief_to_var(var, factor)

    def get_belief_from_var(
        self, var: str, factor: str
    ) -> Optional[npt.NDArray[np.float64]]:
        r"""Returns the current belief from var to factor.

        Parameters
        ----------
        var : string
            Identifier of the variable for which distribution must be returned.
        factor : string
            Identifier of the factor for which distribution must be returned.

        Returns
        -------
        distribution : array_like, f64
            Belief on the edge from `var` to `factor`. If `factor` is SINGLE, distribution has shape
            `(nc)`. Else, it has shape `(n,nc)`.
            If the belief is a uniform distribution, None may be returned
            (but this is not guaranteed).
        """
        return self._inner.get_belief_from_var(var, factor)

    def propagate_var(self, var: str):
        """Run belief propagation on variable var.

        This fetches beliefs from adjacent factors, computes the var
        distribution, and sends updated beliefs to all adjacent factors.

        Parameters
        ----------
        var : string
            Identifier of the variable.

        """
        return self._inner.propagate_var(var)

    def propagate_factor(self, factor: str):
        """Run belief propagation on the given factor.

        This fetches beliefs from adjacent variables and sends updated beliefs
        to all adjacent variables.

        Parameters
        ----------
        var : string
            Identifier of the variable.

        """
        return self._inner.propagate_factor_all(factor)

    def debug(self):
        """Debug-print the current state."""
        s = []
        s.append("VAR DISTRIBUTION")
        for var in self._inner.graph().var_names():
            s.append(f"\tVar {var}")
            s.append(repr(self.get_distribution(var)))
        s.append("FACTORS FROM VARS")
        for factor in self._inner.graph().factor_names():
            for var in self._inner.graph().factor_scope(factor):
                s.append(f"\t{var} -> {factor}")
                s.append(repr(self.get_belief_from_var(var, factor)))
        s.append("FACTORS TO VARS")
        for factor in self._inner.graph().factor_names():
            for var in self._inner.graph().factor_scope(factor):
                s.append(f"\t{factor} -> {var}")
                s.append(repr(self.get_belief_to_var(var, factor)))
        return "\n".join(s)
