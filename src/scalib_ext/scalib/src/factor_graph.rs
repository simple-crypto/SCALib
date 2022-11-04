use crate::bp_compute::Distribution;
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

pub struct FactorGraph {
    nc: usize,
    vars: NamedList<Var>,
    factors: Vec<Factor>,
    edges: Vec<Edge>,
    publics: NamedList<Public>,
    tables: NamedList<Table>,
}

#[derive(Debug, Clone)]
pub enum PublicValue {
    Single(ClassVal),
    Multi(Vec<ClassVal>),
}
impl PublicValue {
  pub fn as_slice(&self) -> &[ClassVal] {
    match self {
      PublicValue::Single(x) => std::slice::from_ref(x),
      PublicValue::Multi(x) => x.as_slice(),
    }
  }
}

pub struct BPState {
    graph: std::rc::Rc<FactorGraph>,
    nmulti: u32,
    public_values: Vec<PublicValue>,
    // public value for each function node
    pub_reduced: Vec<PublicValue>,
    // evidence for each var
    evidence: Vec<Distribution>,
    // current proba for each var
    var_state: Vec<Distribution>,
    // beliefs on each edge
    belief_from_var: Vec<Distribution>,
    belief_to_var: Vec<Distribution>,
}

#[derive(Debug, Clone)]
pub enum BPError {
    WrongDistributionKind,
    MissingEdge,
}

impl FactorGraph {
    pub fn new() -> Self {
        todo!()
    }
    pub fn edge(&self, var: VarId, factor: FactorId) -> Option<EdgeId> {
        self.vars[var].edges.get_index(factor).map(|(_, e)| *e)
    }
}

impl BPState {
    pub fn new(graph: std::rc::Rc<FactorGraph>, nmulti: u32, public_values: Vec<PublicValue>) -> Self {
        let var_state: Vec<_> = graph
            .vars
            .values()
            .map(|v| Distribution::new(v.multi, graph.nc, nmulti))
            .collect();
        let beliefs: Vec<_> = graph
            .edges
            .iter()
            .map(|e| Distribution::new(graph.factors[e.factor].multi, graph.nc, nmulti))
            .collect();
        let pub_reduced = Self::reduce_pub(&graph, &public_values);
        Self {
            evidence: var_state.clone(),
            belief_from_var: beliefs.clone(),
            belief_to_var: beliefs,
            var_state,
            graph,
            nmulti,
            public_values,
            pub_reduced,
        }
    }
    fn reduce_pub(graph: &FactorGraph, public_values: &[PublicValue]) -> Vec<PublicValue> {
      todo!()
    }
    pub fn is_cyclic(&self) -> bool {
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
    pub fn set_evidence(&mut self, var: VarId, evidence: Distribution) -> Result<(), BPError> {
        if self.graph.vars[var].multi != evidence.multi() {
            Err(BPError::WrongDistributionKind)
        } else {
            self.evidence[var] = evidence;
            Ok(())
        }
    }
    pub fn drop_evidence(&mut self, var: VarId) {
        self.evidence[var] = self.evidence[var].as_uniform();
    }
    pub fn get_state(&self, var: VarId) -> &Distribution {
        &self.var_state[var]
    }
    pub fn set_state(&mut self, var: VarId, state: Distribution) -> Result<(), BPError> {
        if self.graph.vars[var].multi != state.multi() {
            Err(BPError::WrongDistributionKind)
        } else {
            self.var_state[var] = state;
            Ok(())
        }
    }
    pub fn get_belief_to_var(&self, var: VarId, factor: FactorId) -> Result<&Distribution, BPError> {
        self.graph
            .edge(var, factor)
            .map(|e| &self.belief_to_var[e])
            .ok_or(BPError::MissingEdge)
    }
    pub fn get_belief_from_var(&self, var: VarId, factor: FactorId) -> Result<&Distribution, BPError> {
        self.graph
            .edge(var, factor)
            .map(|e| &self.belief_from_var[e])
            .ok_or(BPError::MissingEdge)
    }
    // Propagation type:
    // belief to var -> var
    // var -> belief to func
    // trhough func: towards all vars, towards a subset of vars
    pub fn propagate_to_var(&mut self, var: VarId) {
        let distr_iter = self.graph.vars[var] .edges .values() .map(|e| &self.belief_to_var[*e]);
        self.var_state[var].reset();
        self.var_state[var] = self.evidence[var].clone();
        self.var_state[var].multiply(distr_iter);
    }
    pub fn propagate_from_var(&mut self, edge: EdgeId) {
        let var = self.graph.edges[edge].var;
        self.belief_from_var[edge].reset();
        self.belief_from_var[edge] = Distribution::divide(
            &self.var_state[var],
            &self.belief_to_var[edge],
        );
    }
    pub fn propagate_factor(&mut self, factor_id: FactorId, dest: &[VarId]) {
        let factor = &self.graph.factors[factor_id];
        // Pre-erase to have buffers available in cache allocator.
        for d in dest {
            self.belief_to_var[factor.edges[d]].reset();
        }
        // Use a macro to call very similar functions in match arms.
        // Needed because of anonymous return types of these functions.
        macro_rules! prop_factor {
            ($f:ident, $($arg:expr),*) => {
                {
                    let it = $f(factor, &self.belief_from_var, dest, $($arg,)*);
                    for (distr, dest) in it.zip(dest.iter()) {
                        self.belief_to_var[factor.edges[dest]]= distr;
                    }
                }
            };
        }
        match factor.kind {
            FactorKind::AND => prop_factor!(factor_gen_and, &self.pub_reduced[factor_id], false, false),
            FactorKind::NAND => prop_factor!(factor_gen_and, &self.pub_reduced[factor_id], false, true),
            FactorKind::OR => prop_factor!(factor_gen_and, &self.pub_reduced[factor_id], true, true),
            FactorKind::NOR => prop_factor!(factor_gen_and, &self.pub_reduced[factor_id], true, false),
            FactorKind::XOR => prop_factor!(factor_xor, &self.pub_reduced[factor_id]),
            FactorKind::NOT => prop_factor!(factor_not,),
            FactorKind::ADD => prop_factor!(factor_add, &self.pub_reduced[factor_id]),
            FactorKind::MUL => prop_factor!(factor_mul, &self.pub_reduced[factor_id]),
            FactorKind::LOOKUP { table } => prop_factor!(factor_lookup, &self.graph.tables[table]),
        }
    }

    // Higher-level
    pub fn propagate_factor_all(&mut self, factor: FactorId) { todo!() }
    pub fn propagate_var_all(&mut self, var: VarId) { todo!() }
    pub fn propagate_loopy_step(&mut self) { todo!() }
    pub fn propagate_full(&mut self) { todo!() }
}


fn factor_gen_and<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    pub_red: &PublicValue,
    invert_op: bool,
    invert_res: bool,
) -> impl Iterator<Item=Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}

fn factor_xor<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    pub_red: &PublicValue,
) -> impl Iterator<Item=Distribution> + 'a {
    let mut acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
    for d in factor.edges.values().map(|e| &belief_from_var[*e]) {
        // TODO re-use a bufer...
        let mut d = d.clone();
        d.wht();
        acc.multiply(Some(&d).into_iter());
    }
      dest.iter().map(move |dest| {
          let mut d = Distribution::divide(&acc, &belief_from_var[factor.edges[dest]]);
          d.iwht();
          d.regularize();
          d
      })
}

fn factor_not<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
) -> impl Iterator<Item=Distribution> + 'a {
    let in_distr = &belief_from_var[factor.edges[0]];
    let mut res = in_distr.clone();
    res.not();
    return std::iter::once(res);
}

// TODO handle subraction too
fn factor_add<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    pub_red: &PublicValue,
) -> impl Iterator<Item=Distribution> + 'a {
    let mut acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
    for d in factor.edges.values().map(|e| &belief_from_var[*e]) {
        // TODO re-use a bufer...
        let mut d = d.clone();
        d.fft();
        acc.multiply(Some(&d).into_iter());
    }
      dest.iter().map(move |dest| {
          let mut d = Distribution::divide(&acc, &belief_from_var[factor.edges[dest]]);
          d.ifft();
          d.regularize();
          d
      })
}

fn factor_mul<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    pub_red: &PublicValue,
) -> impl Iterator<Item=Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}

fn factor_lookup<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    table: &Table,
) -> impl Iterator<Item=Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}
