use crate::bp_compute::{self, Distribution};
use indexmap::IndexMap;
use itertools::Itertools;

// TODO improvements
// - use a pool for Distribution allocations (can be a simple Vec storing them), to avoid frequent
// allocations

type NamedList<T> = IndexMap<String, T>;

type ClassVal = u32;

type VarId = usize;
type FactorId = usize;
type EdgeId = usize;
type PublicId = usize;
type TableId = usize;

#[derive(Debug, Clone)]
struct Var {
    multi: bool,
    profiled: bool,
    edges: IndexMap<FactorId, EdgeId>,
}

#[derive(Debug, Clone)]
struct Factor {
    kind: FactorKind,
    multi: bool,
    // res is first element, operands come next
    edges: IndexMap<VarId, EdgeId>,
    // May not be allowed for all factor kinds
    publics: Vec<PublicId>,
}

#[derive(Debug, Clone)]
enum FactorKind {
    AND,
    NAND,
    OR,
    NOR,
    XOR,
    XNOR,
    NOT,
    ADD,
    MUL,
    LOOKUP { table: TableId },
}

#[derive(Debug, Clone)]
struct Edge {
    var: VarId,
    pos_var: usize,
    factor: FactorId,
    pos_factor: usize,
}

#[derive(Debug, Clone)]
struct Public {
    multi: bool,
}

#[derive(Debug, Clone)]
struct Table {
    values: Vec<ClassVal>,
}

struct FactorGraph {
    nc: usize,
    vars: NamedList<Var>,
    factors: Vec<Factor>,
    edges: Vec<Edge>,
    publics: NamedList<Public>,
    tables: NamedList<Table>,
}

#[derive(Debug, Clone)]
enum PublicValue {
    Single(ClassVal),
    Multi(Vec<ClassVal>),
}

#[derive(Debug, Clone)]
struct Belief {
    from_var: Distribution,
    to_var: Distribution,
}

impl Belief {
    fn new(multi: bool, nc: usize, nmulti: u32) -> Self {
        Self {
            from_var: Distribution::new(multi, nc, nmulti),
            to_var: Distribution::new(multi, nc, nmulti),
        }
    }
}

struct BPState {
    graph: std::rc::Rc<FactorGraph>,
    nmulti: u32,
    public_values: Vec<PublicValue>,
    // evidence for each var
    evidence: Vec<Distribution>,
    // current proba for each var
    var_state: Vec<Distribution>,
    // beliefs on each edge
    beliefs: Vec<Belief>,
}

#[derive(Debug, Clone)]
enum BPError {
    WrongDistributionKind,
    MissingEdge,
}

impl FactorGraph {
    fn new() -> Self {
        todo!()
    }
    fn edge(&self, var: VarId, factor: FactorId) -> Option<EdgeId> {
        self.vars[var].edges.get_index(factor).map(|(_, e)| *e)
    }
}

impl BPState {
    fn new(graph: std::rc::Rc<FactorGraph>, nmulti: u32, public_values: Vec<PublicValue>) -> Self {
        let var_state: Vec<_> = graph
            .vars
            .values()
            .map(|v| Distribution::new(v.multi, graph.nc, nmulti))
            .collect();
        let beliefs = graph
            .edges
            .iter()
            .map(|e| Belief::new(graph.factors[e.factor].multi, graph.nc, nmulti))
            .collect();
        Self {
            evidence: var_state.clone(),
            beliefs,
            var_state,
            graph,
            nmulti,
            public_values,
        }
    }
    fn is_cyclic(&self) -> bool {
        // Let's do something simple here, and revisit it when we need more sophisticated queries.
        // The factor graph is cyclic if either
        // 1. there is a cycle in a single execution, or
        // 2. two "single" vars are connected by a path that involves a multi node, and nmulti > 1.
        // Special case to avoid further checks
        if self.graph.vars.len() == 0 {
            return false;
        }
        // For 1, we do a DFS walk of the graph starting from an arbitrary var and memoize the vars
        // we've already seen. If we see again a node, there is a cyclce.
        let mut seen_vars = vec![false; self.graph.vars.len()];
        let mut visit_stack = vec![0];
        while let Some(var_id) = visit_stack.pop() {
            if seen_vars[var_id] {
                return true;
            }
            seen_vars[var_id] = true;
            // Enumerate over all incident edges, each edge giving a factor,
            // then we iter over all adjacent factors
            for edge_id in self.graph.vars[var_id].edges.keys() {
                let factor_id = self.graph.edges[*edge_id].factor;
                visit_stack.extend(self.graph.factors[factor_id].edges.keys());
            }
        }
        // For 2., we do the same, but consider all "single" nodes as one:
        // we start from all the "single" nodes together, we ignore paths that touch only "single"
        // node (i.e., the !multi factors), and run the DFS.
        let mut seen_vars = vec![false; self.graph.vars.len()];
        // start from single vars
        let mut visit_stack: Vec<_> = self
            .graph
            .vars
            .values()
            .positions(|var| !var.multi)
            .collect();
        while let Some(var_id) = visit_stack.pop() {
            if seen_vars[var_id] {
                return true;
            }
            seen_vars[var_id] = true;
            // Enumerate over all incident edges, each edge giving a factor,
            // then we iter over all adjacent factors
            for edge_id in self.graph.vars[var_id].edges.keys() {
                let factor_id = self.graph.edges[*edge_id].factor;
                if self.graph.factors[factor_id].multi {
                    visit_stack.extend(self.graph.factors[factor_id].edges.keys());
                }
            }
        }
        return false;
    }
    fn set_evidence(&mut self, var: VarId, evidence: Distribution) -> Result<(), BPError> {
        if self.graph.vars[var].multi != evidence.multi() {
            Err(BPError::WrongDistributionKind)
        } else {
            self.evidence[var] = evidence;
            Ok(())
        }
    }
    fn drop_evidence(&mut self, var: VarId) {
        self.evidence[var] = self.evidence[var].as_uniform();
    }
    fn get_state(&self, var: VarId) -> &Distribution {
        &self.var_state[var]
    }
    fn set_state(&self, var: VarId, state: Distribution) -> Result<(), BPError> {
        if self.graph.vars[var].multi != state.multi() {
            Err(BPError::WrongDistributionKind)
        } else {
            self.var_state[var] = state;
            Ok(())
        }
    }
    fn get_belief(&self, var: VarId, factor: FactorId) -> Result<&Belief, BPError> {
        self.graph
            .edge(var, factor)
            .map(|e| &self.beliefs[e])
            .ok_or(BPError::MissingEdge)
    }
    // Propagation type:
    // belief to var -> var
    // var -> belief to func
    // trhough func: towards all vars, towards a subset of vars
    fn propagate_to_var(&mut self, var: VarId) {
        let multi = self.graph.vars[var].multi;
        let distr_iter = std::iter::once(&self.evidence[var]).chain(
            self.graph.vars[var]
                .edges
                .values()
                .map(|e| &self.beliefs[*e].to_var),
        );
        bp_compute::multiply_distr(&mut self.var_state[var], distr_iter);
    }
    fn propagate_from_var(&mut self, edge: EdgeId) {
        let var = self.graph.edges[edge].var;
        bp_compute::divide_distr(
            &mut self.beliefs[edge].from_var,
            &self.var_state[var],
            &self.beliefs[edge].to_var,
        );
    }
    fn propagate_factor(&mut self, factor: FactorId, dest: &[VarId]) {
        let factor = &self.graph.factors[factor];
        for d in dest {
            self.beliefs[factor.edges[d]].to_var.reset();
        }
        let in_distr = factor.edges.values().map(|e| &self.beliefs[*e].from_var);
        let res_distr = match factor.kind {
            FactorKind::AND => factor_gen_and(factor, in_distr, dest, false, false),
            FactorKind::NAND => factor_gen_and(factor, in_distr, dest, false, true),
            FactorKind::OR => factor_gen_and(factor, in_distr, dest, true, true),
            FactorKind::NOR => factor_gen_and(factor, in_distr, dest, true, false),
            FactorKind::XOR => factor_gen_xor(factor, in_distr, dest, false),
            FactorKind::XNOR => factor_gen_xor(factor, in_distr, dest, true),
            FactorKind::NOT => factor_not(factor, in_distr, dest),
            FactorKind::ADD => factor_add(factor, in_distr, dest),
            FactorKind::MUL => factor_mul(factor, in_distr, dest),
            FactorKind::LOOKUP { table } => {
                factor_lookup(factor, in_distr, dest, &self.graph.tables[table])
            }
        };
        for (distr, dest) in res_distr.into_iter().zip(dest.iter()) {
            self.beliefs[factor.edges[dest]].to_var = distr;
        }
    }

    // Higher-level
    fn propagate_factor_all(&mut self, factor: FactorId) {}
    fn propagate_var_all(&mut self, var: VarId) {}
    fn propagate_loopy_step(&mut self) {}
    fn propagate_full(&mut self) {}
}

fn factor_gen_and<'a>(
    factor: &Factor,
    in_distr: impl Iterator<Item = &'a Distribution>,
    dest: &[VarId],
    invert_op: bool,
    invert_res: bool,
) -> Vec<Distribution> {
    todo!()
}

fn factor_gen_xor<'a>(
    factor: &Factor,
    in_distr: impl Iterator<Item = &'a Distribution>,
    dest: &[VarId],
    invert: bool,
) -> Vec<Distribution> {
    let mut in_distr = in_distr.peekable();
    let mut acc = in_distr.peek().as_cst(0);
    for d in in_distr {
        // TODO re-use a bufer...
        let mut d = d.clone();
        d.wht();
        acc.multiply(Some(d).into_iter());
    }
    acc.iwht();
    todo!()
}

fn factor_not<'a>(
    factor: &Factor,
    in_distr: impl Iterator<Item = &'a Distribution>,
    dest: &[VarId],
) -> Vec<Distribution> {
    let in_distr = in_distr.next().unwrap();
    let mut res = in_distr.clone();
    res.not();
    return vec![res];
}

// TODO handle subraction too
fn factor_add<'a>(
    factor: &Factor,
    in_distr: impl Iterator<Item = &'a Distribution>,
    dest: &[VarId],
) -> Vec<Distribution> {
    todo!()
}

fn factor_mul<'a>(
    factor: &Factor,
    in_distr: impl Iterator<Item = &'a Distribution>,
    dest: &[VarId],
) -> Vec<Distribution> {
    todo!()
}

fn factor_lookup<'a>(
    factor: &Factor,
    in_distr: impl Iterator<Item = &'a Distribution>,
    dest: &[VarId],
    table: &Table,
) -> Vec<Distribution> {
    todo!()
}
