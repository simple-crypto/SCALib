
use itertools::Itertools;
use super::{Distribution, PublicValue, FactorGraph};
use super::factor_graph::{Table,Factor, EdgeId, VarId, FactorId, FactorKind};
use super::ClassVal;

// TODO improvements
// - use a pool for Distribution allocations (can be a simple Vec storing them), to avoid frequent
// allocations

pub struct BPState {
    graph: std::rc::Rc<FactorGraph>,
    nmulti: u32,
    // one public for every factor. Set to 0 if not relevant.
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

impl BPState {
    pub fn new(
        graph: std::rc::Rc<FactorGraph>,
        nmulti: u32,
        public_values: Vec<PublicValue>,
    ) -> Self {
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
        fn merge_pubs<F: Fn(ClassVal, ClassVal) -> ClassVal>(f: F, p1: PublicValue, p2: &PublicValue) -> PublicValue {
            match (p1, p2) {
                (PublicValue::Single(c1), PublicValue::Single(c2)) => PublicValue::Single(f(c1, *c2)),
                (PublicValue::Single(c1), PublicValue::Multi(c2)) => PublicValue::Multi(c2.iter().map(|c2| f(c1, *c2)).collect()),
                (PublicValue::Multi(mut c1), PublicValue::Single(c2)) => {
                    for c1 in c1.iter_mut() {
                        *c1 = f(*c1, *c2);
                    }
                    PublicValue::Multi(c1)
                },
                (PublicValue::Multi(mut c1), PublicValue::Multi(c2)) => {
                    for (c1, c2) in c1.iter_mut().zip(c2.iter()) {
                        *c1 = f(*c1, *c2);
                    }
                    PublicValue::Multi(c1)
                }
            }
        }
        graph.factors.iter().map(|factor| {
            let merge_fn = |a, b| match factor.kind {
                FactorKind::AND { vars_neg: _ } => a & b,
                FactorKind::OR { vars_neg: _ } => a | b,
                FactorKind::XOR => a ^ b,
                FactorKind::ADD => (((a as usize) + (b as usize)) % graph.nc) as ClassVal,
                FactorKind::MUL => (((a as u64) + (b as u64)) % (graph.nc as u64)) as ClassVal,
                FactorKind::NOT | FactorKind::LOOKUP { .. } => unreachable!(),
            };
            let init = match factor.kind {
                FactorKind::AND { vars_neg: _ } => PublicValue::Single((graph.nc -1) as ClassVal),
                FactorKind::OR { vars_neg: _ } | FactorKind::XOR | FactorKind::ADD => PublicValue::Single(0),
                FactorKind::MUL => PublicValue::Single(1),
                FactorKind::NOT | FactorKind::LOOKUP { .. } => unreachable!(),
            };
            factor.publics.iter().map(|pub_id| &public_values[*pub_id]).fold(init, |p1, p2| merge_pubs(merge_fn, p1, p2))
        }).collect()
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
        // we've already seen. If we see again a node, there is a cycle.
        // We start from all not-yet expored vars to cover all connected components.
        let mut seen_vars = vec![false; self.graph.vars.len()];
        let mut visit_stack = vec![];
        for start_var in 0..self.graph.vars.len() {
            if !seen_vars[start_var] {
                visit_stack.push(start_var);
            }
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
        }
        if self.nmulti == 1 {
            return true;
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
    pub fn get_belief_to_var(
        &self,
        var: VarId,
        factor: FactorId,
    ) -> Result<&Distribution, BPError> {
        self.graph
            .edge(var, factor)
            .map(|e| &self.belief_to_var[e])
            .ok_or(BPError::MissingEdge)
    }
    pub fn get_belief_from_var(
        &self,
        var: VarId,
        factor: FactorId,
    ) -> Result<&Distribution, BPError> {
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
        let distr_iter = self.graph.vars[var]
            .edges
            .values()
            .map(|e| &self.belief_to_var[*e]);
        self.var_state[var].reset();
        self.var_state[var] = self.evidence[var].clone();
        self.var_state[var].multiply(distr_iter);
    }
    pub fn propagate_from_var(&mut self, edge: EdgeId) {
        let var = self.graph.edges[edge].var;
        self.belief_from_var[edge].reset();
        self.belief_from_var[edge] =
            Distribution::divide(&self.var_state[var], &self.belief_to_var[edge]);
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
                    let it = $f(factor, &mut self.belief_from_var, dest, $($arg,)*);
                    for (distr, dest) in it.zip(dest.iter()) {
                        self.belief_to_var[factor.edges[dest]]= distr;
                    }
                }
            };
        }
        match &factor.kind {
            FactorKind::AND { vars_neg } => {
                prop_factor!(factor_gen_and, &self.pub_reduced[factor_id], vars_neg.as_slice(), false)
            }
            FactorKind::OR { vars_neg } => {
                prop_factor!(factor_gen_and, &self.pub_reduced[factor_id], vars_neg.as_slice(), true)
            }
            // TODO know when to erase incoming
            FactorKind::XOR => prop_factor!(factor_xor, &self.pub_reduced[factor_id], false),
            FactorKind::NOT => prop_factor!(factor_not,),
            FactorKind::ADD => prop_factor!(factor_add, &self.pub_reduced[factor_id]),
            FactorKind::MUL => prop_factor!(factor_mul, &self.pub_reduced[factor_id]),
            FactorKind::LOOKUP { table } => prop_factor!(factor_lookup, &self.graph.tables[*table]),
        }
    }

    // Higher-level
    pub fn propagate_factor_all(&mut self, factor: FactorId) {
        todo!()
    }
    pub fn propagate_var_all(&mut self, var: VarId) {
        todo!()
    }
    pub fn propagate_loopy_step(&mut self) {
        todo!()
    }
    pub fn propagate_full(&mut self) {
        todo!()
    }
}

fn factor_gen_and<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    pub_red: &PublicValue,
    invert_op: &'a [bool],
    invert_all: bool,
) -> impl Iterator<Item = Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}

fn reset_incoming(
    factor: &Factor,
    belief_from_var: &mut [Distribution],
    dest_taken: &[bool],
    clear_incoming: bool,
) {
    // Everything will be uniform.
    // Clear incoming and reset outgoing.
    if clear_incoming {
        for (taken, e) in dest_taken.iter().zip(factor.edges.values()) {
            if *taken {
                belief_from_var[*e].reset();
            }
        }
    }
}

fn factor_xor<'a>(
    factor: &'a Factor,
    belief_from_var: &'a mut [Distribution],
    dest: &'a [VarId],
    pub_red: &PublicValue,
    clear_incoming: bool,
) -> impl Iterator<Item = Distribution> + 'a {
    // TODO special case for single-input case.
    let mut acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
    acc.wht();
    let mut taken_dest = vec![false; factor.edges.len()];
    for dest in dest {
        taken_dest[factor.edges[dest]] = true;
    }
    let mut uniform_iter = factor
        .edges
        .values()
        .zip(taken_dest.iter())
        .enumerate()
        .filter(|(_, (e, _))| !belief_from_var[**e].is_full());
    let uniform_op = uniform_iter.next();
    if let Some((i, (e_dest, t))) = uniform_op {
        if !*t || uniform_iter.next().is_some() {
            // At least 2 uniform operands, or single uniform is not in dest,
            // all dest messages are uniform.
            reset_incoming(factor, belief_from_var, &taken_dest, clear_incoming);
            return vec![acc.as_uniform(); dest.len()].into_iter();
        } else {
            // Single uniform op, only compute for that one.
            for e in factor.edges.values() {
                if e != e_dest {
                    let mut d = if clear_incoming {
                        belief_from_var[*e].reset()
                    } else {
                        belief_from_var[*e].clone()
                    };
                    d.wht();
                    d.make_non_zero_signed();
                    acc.multiply(Some(&d).into_iter());
                }
            }
            acc.wht();
            acc.regularize();
            let mut res = vec![acc.as_uniform(); dest.len()];
            res[i] = acc;
            return res.into_iter();
        }
    } else {
        // Here we have to actually compute.
        // Simply make the product if Walsh-Hadamard domain
        // We do take the product of all factors then divide because some factors could be zero.
        let mut dest_wht = Vec::with_capacity(dest.len());
        for (e, taken) in factor.edges.values().zip(taken_dest.iter()) {
            let mut d = if clear_incoming {
                belief_from_var[*e].reset()
            } else {
                belief_from_var[*e].clone()
            };
            assert!(d.is_full());
            d.wht();
            d.make_non_zero_signed();
            // We either multiply (non-taken distributions) or we add to the vector of factors.
            if !*taken {
                acc.multiply(Some(&d).into_iter());
            } else {
                dest_wht.push(d);
            }
        }
        // This could be done in O(l log l) instead of O(l^2) where l=dest.len()
        // by better caching product computations.
        return (0..dest.len())
            .map(|i| {
                let mut res = acc.clone();
                res.multiply((0..dest.len()).filter(|j| *j != i).map(|j| &dest_wht[j]));
                res.wht();
                res.regularize();
                res
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
}

fn factor_not<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
) -> impl Iterator<Item = Distribution> + 'a {
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
) -> impl Iterator<Item = Distribution> + 'a {
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
) -> impl Iterator<Item = Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}

fn factor_lookup<'a>(
    factor: &'a Factor,
    belief_from_var: &'a [Distribution],
    dest: &'a [VarId],
    table: &Table,
) -> impl Iterator<Item = Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}
