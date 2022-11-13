use itertools::Itertools;
use thiserror::Error;

use super::factor_graph::{
    EdgeId, EdgeSlice, EdgeVec, Factor, FactorId, FactorKind, FactorVec, Table, VarId, VarVec,
};
use super::ClassVal;
use super::{Distribution, FactorGraph, PublicValue};

// TODO improvements
// - use a pool for Distribution allocations (can be a simple Vec storing them), to avoid frequent
// allocations

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BPState {
    graph: std::sync::Arc<FactorGraph>,
    nmulti: u32,
    // one public for every factor. Set to 0 if not relevant.
    public_values: FactorVec<PublicValue>,
    // public value for each factor
    pub_reduced: FactorVec<PublicValue>,
    // evidence for each var
    evidence: VarVec<Distribution>,
    // current proba for each var
    var_state: VarVec<Distribution>,
    // beliefs on each edge
    belief_from_var: EdgeVec<Distribution>,
    belief_to_var: EdgeVec<Distribution>,
}

#[derive(Debug, Clone, Error)]
pub enum BPError {
    #[error("Wrong distribution kind: got {0}, expected {1}.")]
    WrongDistributionKind(&'static str, &'static str),
    #[error("Wrong number of classes for distribution: got {0}, expected {1}.")]
    WrongDistributionNc(usize, usize),
    #[error("Wrong number of traces for distribution: got {0}, expected {1}.")]
    WrongDistributionNmulti(usize, u32),
}

impl BPState {
    pub fn new(
        graph: std::sync::Arc<FactorGraph>,
        nmulti: u32,
        public_values: Vec<PublicValue>,
    ) -> Self {
        let var_state: VarVec<_> = graph
            .vars
            .values()
            .map(|v| Distribution::new(v.multi, graph.nc, nmulti))
            .collect();
        let beliefs: EdgeVec<_> = graph
            .edges
            .iter()
            .map(|e| Distribution::new(graph.factor(e.factor).multi, graph.nc, nmulti))
            .collect();
        let pub_reduced = Self::reduce_pub(&graph, &public_values);
        Self {
            evidence: var_state.clone(),
            belief_from_var: beliefs.clone(),
            belief_to_var: beliefs,
            var_state,
            graph,
            nmulti,
            public_values: FactorVec::from_vec(public_values),
            pub_reduced,
        }
    }
    fn reduce_pub(graph: &FactorGraph, public_values: &[PublicValue]) -> FactorVec<PublicValue> {
        fn merge_pubs<F: Fn(ClassVal, ClassVal) -> ClassVal>(
            f: F,
            p1: PublicValue,
            p2: &PublicValue,
        ) -> PublicValue {
            match (p1, p2) {
                (PublicValue::Single(c1), PublicValue::Single(c2)) => {
                    PublicValue::Single(f(c1, *c2))
                }
                (PublicValue::Single(c1), PublicValue::Multi(c2)) => {
                    PublicValue::Multi(c2.iter().map(|c2| f(c1, *c2)).collect())
                }
                (PublicValue::Multi(mut c1), PublicValue::Single(c2)) => {
                    for c1 in c1.iter_mut() {
                        *c1 = f(*c1, *c2);
                    }
                    PublicValue::Multi(c1)
                }
                (PublicValue::Multi(mut c1), PublicValue::Multi(c2)) => {
                    for (c1, c2) in c1.iter_mut().zip(c2.iter()) {
                        *c1 = f(*c1, *c2);
                    }
                    PublicValue::Multi(c1)
                }
            }
        }
        graph
            .factors
            .values()
            .map(|factor| {
                let merge_fn = |a, b| match factor.kind {
                    FactorKind::AND { vars_neg: _ } => a & b,
                    FactorKind::OR { vars_neg: _ } => a | b,
                    FactorKind::XOR => a ^ b,
                    FactorKind::ADD => (((a as usize) + (b as usize)) % graph.nc) as ClassVal,
                    FactorKind::MUL => (((a as u64) + (b as u64)) % (graph.nc as u64)) as ClassVal,
                    FactorKind::NOT | FactorKind::LOOKUP { .. } => unreachable!(),
                };
                let init = match factor.kind {
                    FactorKind::AND { vars_neg: _ } => {
                        PublicValue::Single((graph.nc - 1) as ClassVal)
                    }
                    FactorKind::OR { vars_neg: _ } | FactorKind::XOR | FactorKind::ADD => {
                        PublicValue::Single(0)
                    }
                    FactorKind::MUL => PublicValue::Single(1),
                    FactorKind::NOT | FactorKind::LOOKUP { .. } => unreachable!(),
                };
                factor
                    .publics
                    .iter()
                    .map(|pub_id| &public_values[*pub_id])
                    .fold(init, |p1, p2| merge_pubs(merge_fn, p1, p2))
            })
            .collect()
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
        let mut seen_vars: VarVec<bool> = self.graph.range_vars().map(|_| false).collect();
        let mut visit_stack = VarVec::new();

        for start_var in self.graph.range_vars() {
            if !seen_vars[start_var] {
                visit_stack.push(start_var);
            }
            while let Some(var_id) = visit_stack.pop() {
                if seen_vars[var_id] {
                    return true;
                }
                seen_vars[var_id] = true;
                // Enumerate over all incident edges, each edge giving a factor,
                // then we iter over all adjacent vars to the factor
                for factor_id in self.graph.var(var_id).edges.keys() {
                    visit_stack.extend(self.graph.factor(*factor_id).edges.keys());
                }
            }
        }
        if self.nmulti == 1 {
            return true;
        }
        // For 2., we do the same, but consider all "single" nodes as one:
        // we start from all the "single" nodes together, we ignore paths that touch only "single"
        // node (i.e., the !multi factors), and run the DFS.
        let mut seen_vars: VarVec<bool> = std::iter::repeat(false)
            .take(self.graph.vars.len())
            .collect();
        // start from single vars
        let mut visit_stack: VarVec<_> = self
            .graph
            .vars
            .values()
            .positions(|var| !var.multi)
            .map(VarId::from_idx)
            .collect();
        while let Some(var_id) = visit_stack.pop() {
            if seen_vars[var_id] {
                return true;
            }
            seen_vars[var_id] = true;
            // Enumerate over all incident edges, each edge giving a factor,
            // then we iter over all adjacent vars to the factor
            for factor_id in self.graph.var(var_id).edges.keys() {
                if self.graph.factor(*factor_id).multi {
                    visit_stack.extend(self.graph.factor(*factor_id).edges.keys());
                }
            }
        }
        return false;
    }
    pub fn get_graph(&self) -> &std::sync::Arc<FactorGraph> {
        &self.graph
    }
    fn check_distribution(&self, distr: &Distribution, multi: bool) -> Result<(), BPError> {
        if distr.multi() != multi {
            Err(BPError::WrongDistributionKind(
                if distr.multi() { "multi" } else { "single" },
                if multi { "multi" } else { "single" },
            ))
        } else if distr.shape().1 != self.graph.nc {
            Err(BPError::WrongDistributionNc(distr.shape().1, self.graph.nc))
        } else if distr.multi() && self.nmulti as usize != distr.shape().0 {
            Err(BPError::WrongDistributionNmulti(
                distr.shape().0,
                self.nmulti,
            ))
        } else {
            Ok(())
        }
    }
    pub fn set_evidence(&mut self, var: VarId, evidence: Distribution) -> Result<(), BPError> {
        self.check_distribution(&evidence, self.graph.var_multi(var))?;
        self.evidence[var] = evidence;
        Ok(())
    }
    pub fn drop_evidence(&mut self, var: VarId) {
        self.evidence[var] = self.evidence[var].as_uniform();
    }
    pub fn get_state(&self, var: VarId) -> &Distribution {
        &self.var_state[var]
    }
    pub fn set_state(&mut self, var: VarId, state: Distribution) -> Result<(), BPError> {
        self.check_distribution(&state, self.graph.var_multi(var))?;
        self.var_state[var] = state;
        Ok(())
    }
    pub fn drop_state(&mut self, var: VarId) {
        self.var_state[var] = self.var_state[var].as_uniform();
    }
    pub fn get_belief_to_var(&self, edge: EdgeId) -> &Distribution {
        &self.belief_to_var[edge]
    }
    pub fn get_belief_from_var(&self, edge: EdgeId) -> &Distribution {
        &self.belief_from_var[edge]
    }
    // Propagation type:
    // belief to var -> var
    // var -> belief to func
    // trhough func: towards all vars, towards a subset of vars
    pub fn propagate_to_var(&mut self, var: VarId) {
        let distr_iter = self
            .graph
            .var(var)
            .edges
            .values()
            .map(|e| &self.belief_to_var[*e]);
        self.var_state[var].reset();
        self.var_state[var] = self.evidence[var].clone();
        self.var_state[var].multiply(distr_iter);
        self.var_state[var].regularize();
    }
    pub fn propagate_from_var(&mut self, edge: EdgeId) {
        // TODO consider not dividing for better stability ?
        // That should be ok if we ensure that there is no zero element and no
        // underflow (or denormalization).
        // This is guaranteed as long as min_proba > var_degree * MIN_POSITIVE
        let var = self.graph.edges[edge].var;
        self.belief_from_var[edge].reset();
        self.belief_from_var[edge] =
            Distribution::divide(&self.var_state[var], &self.belief_to_var[edge]);
    }
    pub fn propagate_factor(&mut self, factor_id: FactorId, dest: &[VarId]) {
        let factor = self.graph.factor(factor_id);
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
                        self.belief_to_var[factor.edges[dest]] = distr;
                    }
                }
            };
        }
        match &factor.kind {
            FactorKind::AND { vars_neg } => {
                prop_factor!(
                    factor_gen_and,
                    &self.pub_reduced[factor_id],
                    vars_neg.as_slice(),
                    false
                )
            }
            FactorKind::OR { vars_neg } => {
                prop_factor!(
                    factor_gen_and,
                    &self.pub_reduced[factor_id],
                    vars_neg.as_slice(),
                    true
                )
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
        let dest: Vec<_> = self.graph.factor(factor).edges.keys().cloned().collect();
        self.propagate_factor(factor, dest.as_slice());
    }
    pub fn propagate_from_var_all(&mut self, var: VarId) {
        for i in 0..self.graph.var(var).edges.len() {
            self.propagate_from_var(self.graph.var(var).edges[i]);
        }
        for i in 0..self.graph.var(var).edges.len() {
            self.belief_to_var[self.graph.var(var).edges[i]].reset();
        }
    }
    pub fn propagate_var(&mut self, var: VarId) {
        self.propagate_to_var(var);
        self.propagate_from_var_all(var);
    }
    pub fn propagate_loopy_step(&mut self, n_steps: u32) {
        for _ in 0..n_steps {
            for var_id in self.graph.range_vars() {
                self.propagate_var(var_id);
            }
            for factor_id in self.graph.range_factors() {
                self.propagate_factor_all(factor_id);
            }
        }
    }
    pub fn propagate_full(&mut self) {
        // for non-cyclic graph
        todo!()
    }
}

fn factor_gen_and<'a>(
    factor: &'a Factor,
    belief_from_var: &'a mut EdgeSlice<Distribution>,
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
    belief_from_var: &mut EdgeSlice<Distribution>,
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
    belief_from_var: &'a mut EdgeSlice<Distribution>,
    dest: &'a [VarId],
    pub_red: &PublicValue,
    clear_incoming: bool,
) -> impl Iterator<Item = Distribution> + 'a {
    // Special case for single-input XOR
    if factor.edges.len() == 2 {
        return (0..2)
            .map(|i| {
                let mut distr = if clear_incoming {
                    belief_from_var[factor.edges[1 - i]].reset()
                } else {
                    belief_from_var[factor.edges[1 - i]].clone()
                };
                distr.xor_cst(pub_red);
                distr
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
    // TODO special case for single-input case.
    let mut acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
    acc.wht();
    let mut taken_dest = vec![false; factor.edges.len()];
    for dest in dest {
        taken_dest[factor.edges.get_index_of(dest).unwrap()] = true;
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
    belief_from_var: &'a EdgeSlice<Distribution>,
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
    belief_from_var: &'a EdgeSlice<Distribution>,
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
    belief_from_var: &'a EdgeSlice<Distribution>,
    dest: &'a [VarId],
    pub_red: &PublicValue,
) -> impl Iterator<Item = Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}

fn factor_lookup<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Distribution>,
    dest: &'a [VarId],
    table: &Table,
) -> impl Iterator<Item = Distribution> + 'a {
    #![allow(unreachable_code)]
    todo!();
    [].into_iter()
}
