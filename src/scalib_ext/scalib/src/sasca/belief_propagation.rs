use std::sync::{Arc, RwLock};

use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};
use thiserror::Error;

use super::factor_graph as fg;
use super::factor_graph::{
    EdgeId, EdgeSlice, EdgeVec, ExprFactor, Factor, FactorGraph, FactorId, FactorKind, FactorVec,
    Node, PublicValue, Table, VarId, VarVec,
};
use super::{ClassVal, Distribution};
use ndarray::s;

// TODO improvements
// - use a pool for Distribution allocations (can be a simple Vec storing them), to avoid frequent
// allocations

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GenFactor {
    Single(GenFactorInner),
    Multi(Vec<GenFactorInner>),
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GenFactorInner {
    Dense(ndarray::ArrayD<f64>),
    SparseFunctional(ndarray::Array2<ClassVal>),
}

// Workaround since the plans are not Serialize of Debug
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "FftPlansSer", into = "FftPlansSer")]
pub(super) struct FftPlans {
    pub(super) size: usize,
    pub(super) r2c: Arc<dyn realfft::RealToComplex<f64>>,
    pub(super) c2r: Arc<dyn realfft::ComplexToReal<f64>>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FftPlansSer {
    size: usize,
}

impl From<FftPlansSer> for FftPlans {
    fn from(p: FftPlansSer) -> Self {
        Self::new(p.size)
    }
}

impl From<FftPlans> for FftPlansSer {
    fn from(p: FftPlans) -> Self {
        Self { size: p.size }
    }
}

impl std::fmt::Debug for FftPlans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("FftPlans")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl FftPlans {
    fn new(size: usize) -> Self {
        let mut planner = realfft::RealFftPlanner::new();
        Self {
            size,
            r2c: planner.plan_fft_forward(size),
            c2r: planner.plan_fft_inverse(size),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BPState {
    graph: std::sync::Arc<FactorGraph>,
    // number of parallel executions for PARA vars
    nmulti: u32,
    // list of public values
    public_values: Vec<PublicValue>,
    // one public value for every factor. Set to 0 if not relevant.
    pub_reduced: FactorVec<PublicValue>,
    // generalized factors values
    gen_factors: Vec<GenFactor>,
    // evidence for each var
    evidence: VarVec<Arc<RwLock<Distribution>>>,
    // current proba for each var
    var_state: VarVec<Arc<RwLock<Distribution>>>,
    // beliefs on each edge
    belief_from_var: EdgeVec<Arc<RwLock<Distribution>>>,
    belief_to_var: EdgeVec<Arc<RwLock<Distribution>>>,
    // save if cyclic
    cyclic: bool,
    // fft plans
    plans: FftPlans,
}

#[derive(Debug, Clone, Error)]
pub enum BPError {
    #[error("Wrong distribution kind: got {0}, expected {1}.")]
    WrongDistributionKind(&'static str, &'static str),
    #[error("Wrong number of classes for distribution: got {0}, expected {1}.")]
    WrongDistributionNc(usize, usize),
    #[error("Wrong number of traces for distribution: got {0}, expected {1}.")]
    WrongDistributionNmulti(usize, u32),
    #[error("The distribution is not is C memory order. Shape: {0:?}, strides: {1:?}.")]
    DistributionLayout(Vec<usize>, Vec<isize>),
    #[error("Cannot run acyclic BP on a cyclic graph.")]
    NotAcyclic,
}

impl BPState {
    pub fn new(
        graph: std::sync::Arc<FactorGraph>,
        nmulti: u32,
        public_values: Vec<PublicValue>,
        gen_factors: Vec<GenFactor>,
    ) -> Self {
        let ev_state: VarVec<_> = graph
            .vars
            .values()
            .map(|v| Arc::new(RwLock::new(Distribution::new(v.multi, graph.nc, nmulti))))
            .collect();
        let var_state: VarVec<_> = graph
            .vars
            .values()
            .map(|v| Arc::new(RwLock::new(Distribution::new(v.multi, graph.nc, nmulti))))
            .collect();
        let beliefs_from: EdgeVec<_> = graph
            .edges
            .iter()
            .map(|e| {
                Arc::new(RwLock::new(Distribution::new(
                    graph.factor(e.factor).multi,
                    graph.nc,
                    nmulti,
                )))
            })
            .collect();
        let beliefs_to: EdgeVec<_> = graph
            .edges
            .iter()
            .map(|e| {
                Arc::new(RwLock::new(Distribution::new(
                    graph.var(e.var).multi,
                    graph.nc,
                    nmulti,
                )))
            })
            .collect();
        let pub_reduced = graph.reduce_pub(&public_values);
        let cyclic = graph.is_cyclic(nmulti > 1);
        let plans = FftPlans::new(graph.nc);
        Self {
            evidence: ev_state,
            belief_from_var: beliefs_from,
            belief_to_var: beliefs_to,
            var_state,
            graph,
            nmulti,
            public_values,
            pub_reduced,
            cyclic,
            plans,
            gen_factors,
        }
    }
    pub fn is_cyclic(&self) -> bool {
        self.cyclic
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
        *self.evidence[var].write().unwrap() = evidence;
        Ok(())
    }
    pub fn drop_evidence(&mut self, var: VarId) {
        *self.evidence[var].write().unwrap() = self.evidence[var].read().unwrap().as_uniform();
    }
    pub fn get_state(&self, var: VarId) -> Arc<RwLock<Distribution>> {
        self.var_state[var].clone()
    }
    pub fn set_state(&mut self, var: VarId, state: Distribution) -> Result<(), BPError> {
        self.check_distribution(&state, self.graph.var_multi(var))?;
        *self.var_state[var].write().unwrap() = state;
        Ok(())
    }
    pub fn drop_state(&mut self, var: VarId) {
        *self.var_state[var].write().unwrap() = self.var_state[var].read().unwrap().as_uniform();
    }
    pub fn get_belief_to_var(&self, edge: EdgeId) -> Arc<RwLock<Distribution>> {
        self.belief_to_var[edge].clone()
    }
    pub fn get_belief_from_var(&self, edge: EdgeId) -> Arc<RwLock<Distribution>> {
        self.belief_from_var[edge].clone()
    }
    pub fn set_belief_from_var(
        &mut self,
        edge: EdgeId,
        belief: Distribution,
    ) -> Result<(), BPError> {
        self.check_distribution(&belief, self.graph.edge_multi(edge))?;
        *self.belief_from_var[edge].write().unwrap() = belief;
        Ok(())
    }
    pub fn set_belief_to_var(&mut self, edge: EdgeId, belief: Distribution) -> Result<(), BPError> {
        self.check_distribution(&belief, self.graph.edge_multi(edge))?;
        *self.belief_to_var[edge].write().unwrap() = belief;
        Ok(())
    }

    /// propgate only go given edges
    fn propagate_var_t_multi(
        &self,
        var_id: VarId,
        to_edges: Vec<EdgeId>,
        other_edges: Vec<EdgeId>,
        clear_evidence: bool,
        clear_beliefs: bool,
    ) {
        let var = self.graph.var(var_id);
        assert!(var.multi);
        let mut base = self.evidence[var_id]
            .write()
            .unwrap()
            .take_or_clone(clear_evidence);
        let beliefs: Vec<Distribution> = other_edges
            .iter()
            .map(|e| self.belief_to_var[*e].read().unwrap().clone())
            .collect();
        base.multiply_norm(beliefs.iter());
        if clear_beliefs {
            for e in other_edges {
                self.belief_to_var[e].write().unwrap().reset();
            }
        }
        let beliefs: Vec<Distribution> = to_edges
            .iter()
            .map(|e| self.belief_to_var[*e].read().unwrap().clone())
            .collect();
        let (var_state, new_beliefs) =
            super::bp_compute::belief_reciprocal_product(base, beliefs.iter());
        for (e, d) in to_edges.iter().zip(new_beliefs.into_iter()) {
            *self.belief_from_var[*e].write().unwrap() = d;
            if clear_beliefs {
                self.belief_to_var[*e].write().unwrap().reset();
            }
        }
        *self.var_state[var_id].write().unwrap() = var_state;
    }

    fn propagate_var_t_single(
        &self,
        var_id: VarId,
        to_edges: Vec<EdgeId>,
        other_edges: Vec<EdgeId>,
        clear_evidence: bool,
        clear_beliefs: bool,
    ) {
        let var = self.graph.var(var_id);
        assert!(!var.multi);
        let mut base = self.evidence[var_id]
            .write()
            .unwrap()
            .take_or_clone(clear_evidence);
        for e in other_edges {
            base.multiply_to_single(&self.belief_to_var[e].read().unwrap());
            if clear_beliefs {
                self.belief_to_var[e].write().unwrap().reset();
            }
        }
        let (global_products, local_products): (Vec<_>, Vec<_>) = to_edges
            .iter()
            .map(|e| {
                self.belief_to_var[*e]
                    .read()
                    .unwrap()
                    .reciprocal_product(self.evidence[var_id].read().unwrap().as_uniform())
            })
            .unzip();
        let (var_state, new_beliefs_global) =
            super::bp_compute::belief_reciprocal_product(base, global_products.iter());
        for ((e, mut local), global) in to_edges
            .iter()
            .zip(local_products.into_iter())
            .zip(new_beliefs_global.into_iter())
        {
            local.multiply_norm(std::iter::once(&global));
            *self.belief_from_var[*e].write().unwrap() = local;
            if clear_beliefs {
                self.belief_to_var[*e].write().unwrap().reset();
            }
        }
        *self.var_state[var_id].write().unwrap() = var_state;
    }

    pub fn propagate_factor(&self, factor_id: FactorId, dest: &[VarId], clear_incoming: bool) {
        let factor = self.graph.factor(factor_id);
        // Pre-erase to have buffers available in cache allocator.
        for d in dest {
            self.belief_to_var[factor.edges[d]].write().unwrap().reset();
        }
        // Use a macro to call very similar functions in match arms.
        // Needed because of anonymous return types of these functions.
        macro_rules! prop_factor {
            ($f:ident, $($arg:expr),*) => {
                {
                    let it = $f(factor, &self.belief_from_var, dest, clear_incoming, $($arg,)*);
                    for (mut distr, dest) in it.zip(dest.iter()) {
                        distr.regularize();
                        *self.belief_to_var[factor.edges[dest]].write().unwrap() = distr;
                    }
                }
            };
        }
        match &factor.kind {
            FactorKind::Assign { expr, .. } => match expr {
                ExprFactor::AND { .. } => {
                    prop_factor!(factor_gen_and, &self.pub_reduced[factor_id])
                }
                ExprFactor::XOR => prop_factor!(factor_xor, &self.pub_reduced[factor_id]),
                ExprFactor::NOT => prop_factor!(factor_not, (self.graph.nc - 1) as u32),
                ExprFactor::ADD => {
                    prop_factor!(factor_add, &self.pub_reduced[factor_id], &self.plans)
                }
                ExprFactor::MUL => prop_factor!(factor_mul, &self.pub_reduced[factor_id]),
                ExprFactor::LOOKUP { table } => {
                    prop_factor!(factor_lookup, &self.graph.tables[*table])
                }
            },
            FactorKind::GenFactor { id, .. } => {
                let gen_factor = &self.gen_factors[*id];
                prop_factor!(
                    factor_gen_factor,
                    gen_factor,
                    self.public_values.as_slice(),
                    self.nmulti as usize,
                    self.graph.nc
                );
            }
        }
    }

    // Higher-level
    pub fn propagate_factor_all(&self, factor: FactorId) {
        let dest: Vec<_> = self.graph.factor(factor).edges.keys().cloned().collect();
        self.propagate_factor(factor, dest.as_slice(), false);
    }
    pub fn propagate_var(&self, var_id: VarId, clear_beliefs: bool) {
        let clear_evidence = false;
        self.propagate_var_to(
            var_id,
            self.graph
                .var(var_id)
                .edges
                .values()
                .cloned()
                .collect::<Vec<_>>(),
            clear_beliefs,
            clear_evidence,
        );
    }
    pub fn propagate_var_to(
        &self,
        var_id: VarId,
        mut to_edges: Vec<EdgeId>,
        clear_beliefs: bool,
        clear_evidence: bool,
    ) {
        let var = self.graph.var(var_id);
        let mut all_edges = var.edges.values().collect::<Vec<_>>();
        all_edges.sort_unstable();
        to_edges.sort_unstable();
        let other_edges = all_edges
            .iter()
            .merge_join_by(to_edges.iter(), |x, y| x.cmp(&y))
            .filter_map(|x| {
                if let itertools::EitherOrBoth::Left(e) = x {
                    Some(*e)
                } else {
                    None
                }
            })
            .cloned()
            .collect::<Vec<_>>();
        if var.multi {
            self.propagate_var_t_multi(
                var_id,
                to_edges,
                other_edges,
                clear_evidence,
                clear_beliefs,
            );
        } else {
            self.propagate_var_t_single(
                var_id,
                to_edges,
                other_edges,
                clear_evidence,
                clear_beliefs,
            );
        }
    }
    pub fn propagate_all_vars(&mut self, clear_beliefs: bool) {
        self.graph
            .range_vars()
            .par_bridge()
            .for_each(|var_id| self.propagate_var(var_id, clear_beliefs));
    }
    pub fn propagate_loopy_step(&mut self, n_steps: u32, clear_beliefs: bool) {
        for _ in 0..n_steps {
            self.graph
                .range_factors()
                .par_bridge()
                .for_each(|factor_id| self.propagate_factor_all(factor_id));
            self.propagate_all_vars(clear_beliefs);
        }
    }
    pub fn propagate_acyclic(
        &mut self,
        var: VarId,
        clear_intermediates: bool,
        clear_evidence: bool,
    ) -> Result<(), BPError> {
        if self.is_cyclic() {
            return Err(BPError::NotAcyclic);
        }
        for (node, parent) in self.graph.propagation_order(var) {
            match node {
                Node::Var(var_id) => {
                    let to_edges = if let Some(dest_factor) = parent {
                        vec![self.graph.var(var_id).edges[&dest_factor.factor().unwrap()]]
                    } else {
                        vec![]
                    };
                    self.propagate_var_to(var_id, to_edges, clear_intermediates, clear_evidence);
                }
                Node::Factor(factor_id) => {
                    let parent_var = parent.unwrap().var().unwrap();
                    self.propagate_factor(factor_id, &[parent_var], clear_intermediates);
                }
            }
        }
        Ok(())
    }
}

fn factor_gen_and<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    pub_red: &PublicValue,
) -> impl Iterator<Item = Distribution> + 'a {
    let FactorKind::Assign {
        expr: ExprFactor::AND { vars_neg },
        has_res,
    } = &factor.kind
    else {
        unreachable!()
    };
    // Special case for single-input AND
    if has_res & (factor.edges.len() == 2) {
        return dest
            .iter()
            .map(|var| {
                let i = factor.edges.get_index_of(var).unwrap();
                let mut distr = belief_from_var[factor.edges[1 - i]]
                    .write()
                    .unwrap()
                    .take_or_clone(clear_incoming);
                if vars_neg[1 - i] {
                    distr.not();
                }
                if i == 0 {
                    // dest is the result of the AND
                    distr.and_cst(pub_red);
                } else {
                    // dest is an operand of the AND, original distr is result
                    distr.inv_and_cst(pub_red);
                }
                if vars_neg[i] {
                    distr.not();
                }
                distr
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
    // Compute a product in the transformed domain
    // direct transform:
    // - if operand: cumt
    // - if result: opandt
    // inverse transform:
    // - if operand: opandt^-1 = opandt
    // - if result: cumt^-1 = cumti
    let mut acc = belief_from_var[factor.edges[0]]
        .read()
        .unwrap()
        .new_constant(pub_red);
    if *has_res {
        // constant is operand
        acc.cumt();
    } else {
        // constant is result
        acc.opandt();
    }
    let mut taken_dest = vec![false; factor.edges.len()];
    for dest in dest {
        taken_dest[factor.edges.get_index_of(dest).unwrap()] = true;
    }
    // Compute the product in transform domain
    // We do not take the product of all factors then divide because some factors could be zero.
    let mut dest_transformed = Vec::with_capacity(dest.len());
    for ((i, e), taken) in factor.edges.values().enumerate().zip(taken_dest.iter()) {
        let mut d = belief_from_var[*e]
            .write()
            .unwrap()
            .take_or_clone(clear_incoming);
        if vars_neg[i] {
            d.not();
        }
        d.ensure_full();
        if *has_res && (i == 0) {
            d.opandt();
        } else {
            d.cumt();
        }
        // We either multiply (non-taken distributions) or we add to the vector of factors.
        if !*taken {
            acc.multiply(Some(&d).into_iter());
        } else {
            dest_transformed.push(d);
        }
    }
    // This could be done in O(l log l) instead of O(l^2) where l=dest.len()
    // by better caching product computations.
    return (0..dest.len())
        .map(|i| {
            let mut res = acc.clone();
            res.multiply(
                (0..dest.len())
                    .filter(|j| *j != i)
                    .map(|j| &dest_transformed[j]),
            );
            // Inverse transform
            if Some(dest[i]) == factor.res_id() {
                res.cumti();
            } else {
                res.opandt();
            }
            if vars_neg[factor.edges.get_index_of(&dest[i]).unwrap()] {
                res.not();
            }
            res.regularize();
            res
        })
        .collect::<Vec<_>>()
        .into_iter();
}

fn reset_incoming(
    factor: &Factor,
    belief_from_var: &EdgeSlice<Arc<RwLock<Distribution>>>,
    dest_taken: &[bool],
    clear_incoming: bool,
) {
    // Everything will be uniform.
    // Clear incoming and reset outgoing.
    if clear_incoming {
        for (taken, e) in dest_taken.iter().zip(factor.edges.values()) {
            if *taken {
                belief_from_var[*e].write().unwrap().reset();
            }
        }
    }
}

fn factor_xor<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    pub_red: &PublicValue,
) -> impl Iterator<Item = Distribution> + 'a {
    // Special case for single-input XOR
    if factor.edges.len() == 2 {
        return dest
            .iter()
            .map(|var| {
                let i = factor.edges.get_index_of(var).unwrap();
                let mut distr = belief_from_var[factor.edges[1 - i]]
                    .write()
                    .unwrap()
                    .take_or_clone(clear_incoming);
                distr.xor_cst(pub_red);
                distr
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
    let mut acc = belief_from_var[factor.edges[0]]
        .read()
        .unwrap()
        .new_constant(pub_red);
    acc.wht();
    let mut taken_dest = vec![false; factor.edges.len()];
    let mut taken_dest_idx = vec![None; factor.edges.len()];
    for (i, dest) in dest.iter().enumerate() {
        taken_dest[factor.edges.get_index_of(dest).unwrap()] = true;
        taken_dest_idx[factor.edges.get_index_of(dest).unwrap()] = Some(i);
    }
    let mut uniform_iter = factor
        .edges
        .values()
        .zip(taken_dest_idx.iter())
        .filter(|(e, _)| !belief_from_var[**e].read().unwrap().is_full());
    let uniform_op = uniform_iter.next();
    if let Some((e_dest, t)) = uniform_op {
        if t.is_none() || uniform_iter.next().is_some() {
            // At least 2 uniform operands, or single uniform is not in dest,
            // all dest messages are uniform.
            reset_incoming(factor, belief_from_var, &taken_dest, clear_incoming);
            return vec![acc.as_uniform(); dest.len()].into_iter();
        } else {
            // Single uniform op, only compute for that one.
            for e in factor.edges.values() {
                if e != e_dest {
                    let mut d = belief_from_var[*e]
                        .write()
                        .unwrap()
                        .take_or_clone(clear_incoming);
                    d.wht();
                    d.make_non_zero_signed();
                    acc.multiply(Some(&d).into_iter());
                }
            }
            acc.wht();
            acc.regularize();
            let mut res = vec![acc.as_uniform(); dest.len()];
            res[t.unwrap()] = acc;
            return res.into_iter();
        }
    } else {
        // Here we have to actually compute.
        // Simply make the product if Walsh-Hadamard domain
        // We do take the product of all factors then divide because some factors could be zero.
        let mut dest_wht = Vec::with_capacity(dest.len());
        for (e, taken) in factor.edges.values().zip(taken_dest.iter()) {
            let mut d = belief_from_var[*e]
                .write()
                .unwrap()
                .take_or_clone(clear_incoming);
            assert!(d.is_full());
            d.wht();
            // TODO remove this ?
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
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    inv_cst: u32,
) -> impl Iterator<Item = Distribution> + 'a {
    factor_xor(
        factor,
        belief_from_var,
        dest,
        clear_incoming,
        &PublicValue::Single(inv_cst),
    )
}

// TODO handle subtraction too (actually, we can re-write it as an addition by moving terms around).
fn factor_add<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    pub_red: &PublicValue,
    plans: &FftPlans,
) -> impl Iterator<Item = Distribution> + 'a {
    // Special case for single-input ADD
    if factor.edges.len() == 2 {
        // FIXME check for negative operand
        return dest
            .iter()
            .map(|var| {
                let i = factor.edges.get_index_of(var).unwrap();
                let mut distr = belief_from_var[factor.edges[1 - i]]
                    .write()
                    .unwrap()
                    .take_or_clone(clear_incoming);
                distr.add_cst(pub_red, i != 0);
                distr
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
    let mut taken_dest = vec![false; factor.edges.len()];
    let mut negated_vars = vec![false; factor.edges.len()];
    negated_vars[0] = true;
    for dest in dest {
        taken_dest[factor.edges.get_index_of(dest).unwrap()] = true;
    }
    let mut uniform_iter = factor
        .edges
        .iter()
        .zip(taken_dest.iter())
        .zip(negated_vars.iter())
        .filter(|(((_, e), _), _)| !belief_from_var[**e].read().unwrap().is_full());
    let uniform_op = uniform_iter.next();
    let uniform_template = belief_from_var[factor.edges[0]]
        .read()
        .unwrap()
        .as_uniform();
    let (nmulti, nc) = uniform_template.shape();
    let mut fft_tmp = ndarray::Array2::zeros((nmulti, nc / 2 + 1));
    let mut acc_fft = ndarray::Array2::zeros((nmulti, nc / 2 + 1));
    let mut acc_fft_init = false;
    let mut fft_scratch = plans.r2c.make_scratch_vec();
    let mut fft_input_scratch = plans.r2c.make_input_vec();
    if let Some((((v_dest, e_dest), t), dest_negated)) = uniform_op {
        if !*t || uniform_iter.next().is_some() {
            // At least 2 uniform operands, or single uniform is not in dest,
            // all dest messages are uniform.
            reset_incoming(factor, belief_from_var, &taken_dest, clear_incoming);
            return vec![uniform_template; dest.len()].into_iter();
        } else {
            // Single uniform op, only compute for that one.
            for (e, negated_var) in factor.edges.values().zip(negated_vars.iter()) {
                if e != e_dest {
                    let negate = !(dest_negated ^ negated_var);
                    belief_from_var[*e].read().unwrap().fft_to(
                        fft_input_scratch.as_mut_slice(),
                        fft_tmp.view_mut(),
                        fft_scratch.as_mut_slice(),
                        plans,
                        negate,
                    );

                    if acc_fft_init {
                        acc_fft *= &fft_tmp;
                    } else {
                        acc_fft.assign(&fft_tmp);
                        acc_fft_init = true;
                    }
                }
                if clear_incoming {
                    belief_from_var[*e].write().unwrap().reset();
                }
            }
        }
        let mut acc = uniform_template.clone();
        let mut fft_scratch = plans.c2r.make_scratch_vec();
        acc.ifft(acc_fft.view_mut(), fft_scratch.as_mut_slice(), plans, false);
        acc.regularize();
        let mut res = vec![uniform_template; dest.len()];
        res[dest.iter().position(|v| v == v_dest).unwrap()] = acc;
        return res.into_iter();
    } else {
        // Here we have to actually compute.
        // Simply make the product if FFT domain
        // We do take the product of all factors then divide because some factors could be zero.
        let mut dest_fft = Vec::with_capacity(dest.len());
        for ((e, taken), negated_var) in factor
            .edges
            .values()
            .zip(taken_dest.iter())
            .zip(negated_vars.iter())
        {
            if *taken {
                let mut fft_e = ndarray::Array2::zeros((nmulti, nc / 2 + 1));
                belief_from_var[*e].read().unwrap().fft_to(
                    fft_input_scratch.as_mut_slice(),
                    fft_e.view_mut(),
                    fft_scratch.as_mut_slice(),
                    plans,
                    *negated_var,
                );

                dest_fft.push(fft_e);
            } else {
                belief_from_var[*e].read().unwrap().fft_to(
                    fft_input_scratch.as_mut_slice(),
                    fft_tmp.view_mut(),
                    fft_scratch.as_mut_slice(),
                    plans,
                    *negated_var,
                );

                if acc_fft_init {
                    acc_fft *= &fft_tmp;
                } else {
                    acc_fft.assign(&fft_tmp);
                    acc_fft_init = true;
                }
            }
            if clear_incoming {
                belief_from_var[*e].write().unwrap().reset();
            }
        }

        // This could be done in O(l) instead of O(l^2) where l=dest.len() by
        // better caching product computations.
        let mut fft_scratch = plans.c2r.make_scratch_vec();
        return (0..dest.len())
            .map(move |i| {
                let mut res = if acc_fft_init {
                    acc_fft.clone()
                } else {
                    ndarray::Array2::ones(acc_fft.raw_dim())
                };

                for (j, fft_op) in dest_fft.iter().enumerate() {
                    if j != i {
                        res *= fft_op;
                    }
                }
                let idx = factor.edges.get_index_of(&dest[i]).unwrap();
                let mut acc = uniform_template.clone();
                acc.ifft(
                    res.view_mut(),
                    fft_scratch.as_mut_slice(),
                    plans,
                    !negated_vars[idx],
                );
                acc.regularize();
                acc
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
}

fn factor_mul<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    pub_red: &'a PublicValue,
) -> impl Iterator<Item = Distribution> + 'a {
    // This is a simple, non-optimized algorithm.
    let mut dest_iter = dest.iter();
    std::iter::from_fn(move || {
        if let Some(var) = dest_iter.next() {
            // We compute the product and not a factor.
            let is_product = factor.edges.get_index_of(var).unwrap() == 0;
            let mut res: Option<Distribution> = None;
            for (v, e) in factor.edges.iter() {
                if v != var {
                    res = Some(if let Some(res) = res {
                        if is_product {
                            res.op_multiply(&belief_from_var[*e].read().unwrap())
                        } else {
                            res.op_multiply_factor(&belief_from_var[*e].read().unwrap())
                        }
                    } else {
                        belief_from_var[*e]
                            .write()
                            .unwrap()
                            .take_or_clone(clear_incoming)
                    });
                }
            }
            Some(if is_product {
                res.unwrap().op_multiply_cst(pub_red)
            } else {
                res.unwrap().op_multiply_cst_factor(pub_red)
            })
        } else {
            if clear_incoming {
                for e in factor.edges.values() {
                    belief_from_var[*e].write().unwrap().reset();
                }
            }
            None
        }
    })
}

fn factor_lookup<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    table: &'a Table,
) -> impl Iterator<Item = Distribution> + 'a {
    // we know that there is no constant involved
    assert_eq!(factor.edges.len(), 2);
    dest.iter().map(move |dest| {
        let i = factor.edges.get_index_of(dest).unwrap();
        let distr = belief_from_var[factor.edges[1 - i]].clone();
        let res = if i == 0 {
            // dest is res
            distr.read().unwrap().map_table(table.values.as_slice())
        } else {
            distr.read().unwrap().map_table_inv(table.values.as_slice())
        };
        if clear_incoming {
            belief_from_var[factor.edges[1 - i]]
                .write()
                .unwrap()
                .reset();
        }
        res
    })
}

fn factor_gen_factor<'a>(
    factor: &'a Factor,
    belief_from_var: &'a EdgeSlice<Arc<RwLock<Distribution>>>,
    dest: &'a [VarId],
    clear_incoming: bool,
    gen_factor: &'a GenFactor,
    public_values: &'a [PublicValue],
    nmulti: usize,
    nc: usize,
) -> impl Iterator<Item = Distribution> + 'a {
    let fg::FactorKind::GenFactor { operands, .. } = &factor.kind else {
        unreachable!()
    };
    let res: Vec<Distribution> = dest.iter().map(|dest| {
        let dest_idx = factor.edges.get_index_of(dest).unwrap();
        let mut distr = belief_from_var[factor.edges[dest_idx]].read().unwrap().clone();
        distr.ensure_full();
        for i in 0..nmulti {
            let gen_factor = match gen_factor {
                GenFactor::Single(x) => x,
                GenFactor::Multi(x) => &x[i],
            };
            match gen_factor {
                GenFactorInner::Dense(gen_factor) => {
                    assert_eq!(gen_factor.shape().len(), operands.len());
                    // First slice the array with the constants.
                    let gen_factor = gen_factor.slice_each_axis(|ax| match operands[ax.axis.index()] {
                        fg::GenFactorOperand::Var(_, _) => ndarray::Slice::new(0, None, 1),
                        fg::GenFactorOperand::Pub(pub_idx) => {
                            let mut pub_val = public_values[factor.publics[pub_idx].0].get(i) as isize;
                            if factor.publics[pub_idx].1 {
                                if nc.is_power_of_two() {
                                    pub_val = !pub_val;
                                } else {
                                    // TODO Check that we enforce this at graph creation time and return a proper error.
                                    panic!("Cannot negate operands with non-power-of-two number of classes.");
                                }
                            }
                            ndarray::Slice::new(pub_val, Some(pub_val+1), 1)
                        }
                    });
                    let mut gen_factor = gen_factor.to_owned();
                    for (op_idx, op) in operands.iter().enumerate() {
                        if op_idx != dest_idx {
                            if let fg::GenFactorOperand::Var(var_idx, neg) = op {
                                if *neg {
                                    todo!("Negated operands on generalized factors not yet implemented.");
                                }
                                let distr = &belief_from_var[factor.edges[*var_idx]];
                                let mut new_gen_factor: ndarray::ArrayD<f64> = ndarray::ArrayD::zeros(gen_factor.slice_axis(ndarray::Axis(op_idx), ndarray::Slice::new(0, Some(1), 1)).shape());
                                if let Some(distr) = distr.read().unwrap().value() {
                                    for (d, gf) in distr.slice(s![i,..]).iter().zip(gen_factor.axis_chunks_iter(ndarray::Axis(op_idx), 1)) {
                                        new_gen_factor.scaled_add(*d, &gf);
                                    }
                                } else {
                                    for gf in gen_factor.axis_chunks_iter(ndarray::Axis(op_idx), 1) {
                                        new_gen_factor += &gf;
                                    }
                                }
                                gen_factor = new_gen_factor;
                            }
                        }
                    }
                    // Drop useless axes.
                    for _ in 0..dest_idx {
                        gen_factor.index_axis_inplace(ndarray::Axis(0), 0);
                    }
                    for _ in (dest_idx+1)..operands.len() {
                        gen_factor.index_axis_inplace(ndarray::Axis(1), 0);
                    }
                    distr.value_mut().unwrap().slice_mut(s![i,..]).assign(&gen_factor);
                }
                GenFactorInner::SparseFunctional(gen_factor) => {
                    assert_eq!(gen_factor.shape()[1], operands.len());
                    let mut dest_all = distr.value_mut().unwrap();
                    let mut dest = dest_all.slice_mut(s![i,..]);
                    dest.fill(0.0);
                    for op_values in gen_factor.outer_iter() {
                        let mut res = 1.0;
                        for (op_idx, (op, val)) in operands.iter().zip(op_values.iter()).enumerate() {
                            if op_idx != dest_idx {
                                match op {
                                    fg::GenFactorOperand::Var(var_idx, neg) => {
                                        let mut val = *val;
                                        if *neg {
                                            if nc.is_power_of_two() {
                                                val = !val & ((nc - 1) as ClassVal);
                                            } else {
                                                // TODO Check that we enforce this at graph creation time and return a proper error.
                                                panic!("Cannot negate operands with non-power-of-two number of classes.");
                                            }
                                        }
                                        let distr = &belief_from_var[factor.edges[*var_idx]];
                                        // For uniform, we implicitly multiply by 1.0
                                        if let Some(distr) = distr.read().unwrap().value() {
                                            res *= distr[(i, val as usize)];
                                        }
                                    }
                                    fg::GenFactorOperand::Pub(pub_idx) => {
                                        let mut pub_val = public_values[factor.publics[*pub_idx].0].get(i);
                                        if factor.publics[*pub_idx].1 {
                                            if nc.is_power_of_two() {
                                                pub_val = !pub_val & ((nc - 1) as ClassVal);
                                            } else {
                                                // TODO Check that we enforce this at graph creation time and return a proper error.
                                                panic!("Cannot negate operands with non-power-of-two number of classes.");
                                            }
                                        }
                                        if pub_val != *val {
                                            res = 0.0;
                                        }
                                    }
                                }
                            }
                        }
                        dest[op_values[dest_idx] as usize] += res;
                    }
                }
            }
        }
        distr
    }).collect();
    if clear_incoming {
        for e in factor.edges.values() {
            belief_from_var[*e].write().unwrap().reset();
        }
    }
    res.into_iter()
}
