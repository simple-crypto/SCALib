use std::ops::MulAssign;
use std::sync::Arc;

use itertools::Itertools;
use ndarray::{s, Array2, ShapeBuilder};
use thiserror::Error;

use crate::sasca::factor_graph::GenFactorOperand;

use super::factor_graph as fg;
use super::factor_graph::{
    EdgeId, EdgeSlice, EdgeVec, ExprFactor, Factor, FactorGraph, FactorId, FactorKind, FactorVec,
    Node, PublicValue, Table, VarId, VarVec,
};
use super::{ClassVal, Distribution};
use crate::utils::RangeExt;

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
    evidence: VarVec<Distribution>,
    // current proba for each var
    var_state: VarVec<Distribution>,
    // beliefs on each edge
    belief_from_var: EdgeVec<Distribution>,
    belief_to_var: EdgeVec<Distribution>,
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
        let pub_reduced = graph.reduce_pub(&public_values);
        let cyclic = graph.is_cyclic(nmulti > 1);
        let plans = FftPlans::new(graph.nc);
        Self {
            evidence: var_state.clone(),
            belief_from_var: beliefs.clone(),
            belief_to_var: beliefs,
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
    pub fn set_belief_from_var(
        &mut self,
        edge: EdgeId,
        belief: Distribution,
    ) -> Result<(), BPError> {
        self.check_distribution(&belief, self.graph.edge_multi(edge))?;
        self.belief_from_var[edge] = belief;
        Ok(())
    }
    pub fn set_belief_to_var(&mut self, edge: EdgeId, belief: Distribution) -> Result<(), BPError> {
        self.check_distribution(&belief, self.graph.edge_multi(edge))?;
        self.belief_to_var[edge] = belief;
        Ok(())
    }

    /// propgate only go given edges
    fn propagate_var_t_multi(
        &mut self,
        var_id: VarId,
        to_edges: Vec<EdgeId>,
        other_edges: Vec<EdgeId>,
        clear_evidence: bool,
        clear_beliefs: bool,
    ) {
        let var = self.graph.var(var_id);
        assert!(var.multi);
        let mut base = self.evidence[var_id].take_or_clone(clear_evidence);
        base.multiply_norm(other_edges.iter().map(|e| &self.belief_to_var[*e]));
        if clear_beliefs {
            for e in other_edges {
                self.belief_to_var[e].reset();
            }
        }
        let (var_state, new_beliefs) = super::bp_compute::belief_reciprocal_product(
            base,
            to_edges.iter().map(|e| &self.belief_to_var[*e]),
        );
        for (e, d) in to_edges.iter().zip(new_beliefs.into_iter()) {
            self.belief_from_var[*e] = d;
            if clear_beliefs {
                self.belief_to_var[*e].reset();
            }
        }
        self.var_state[var_id] = var_state;
    }

    fn propagate_var_t_single(
        &mut self,
        var_id: VarId,
        to_edges: Vec<EdgeId>,
        other_edges: Vec<EdgeId>,
        clear_evidence: bool,
        clear_beliefs: bool,
    ) {
        let var = self.graph.var(var_id);
        assert!(!var.multi);
        let mut base = self.evidence[var_id].take_or_clone(clear_evidence);
        for e in other_edges {
            base.multiply_to_single(&self.belief_to_var[e]);
            if clear_beliefs {
                self.belief_to_var[e].reset();
            }
        }
        let (global_products, local_products): (Vec<_>, Vec<_>) = to_edges
            .iter()
            .map(|e| self.belief_to_var[*e].reciprocal_product(self.evidence[var_id].as_uniform()))
            .unzip();
        let (var_state, new_beliefs_global) =
            super::bp_compute::belief_reciprocal_product(base, global_products.iter());
        for ((e, mut local), global) in to_edges
            .iter()
            .zip(local_products.into_iter())
            .zip(new_beliefs_global.into_iter())
        {
            local.multiply_norm(std::iter::once(&global));
            self.belief_from_var[*e] = local;
            if clear_beliefs {
                self.belief_to_var[*e].reset();
            }
        }
        self.var_state[var_id] = var_state;
    }

    pub fn propagate_factor(&mut self, factor_id: FactorId, dest: &[VarId], clear_incoming: bool) {
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
                    let it = $f(factor, &mut self.belief_from_var, dest, clear_incoming, $($arg,)*);
                    for (mut distr, dest) in it.zip(dest.iter()) {
                        distr.regularize();
                        self.belief_to_var[factor.edges[dest]] = distr;
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
                ExprFactor::NOT => prop_factor!(
                    factor_not,
                    &self.pub_reduced[factor_id],
                    (self.graph.nc - 1) as u32
                ),
                ExprFactor::ADD { .. } => {
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
    pub fn propagate_factor_all(&mut self, factor: FactorId) {
        let dest: Vec<_> = self.graph.factor(factor).edges.keys().cloned().collect();
        self.propagate_factor(factor, dest.as_slice(), false);
    }
    pub fn propagate_var(&mut self, var_id: VarId, clear_beliefs: bool) {
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
        &mut self,
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
        for var_id in self.graph.range_vars() {
            self.propagate_var(var_id, clear_beliefs);
        }
    }
    pub fn propagate_loopy_step(&mut self, n_steps: u32, clear_beliefs: bool) {
        for _ in 0..n_steps {
            for factor_id in self.graph.range_factors() {
                self.propagate_factor_all(factor_id);
            }
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
    belief_from_var: &'a mut EdgeSlice<Distribution>,
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
                let mut distr = belief_from_var[factor.edges[1 - i]].take_or_clone(clear_incoming);
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
    let mut acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
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
        let mut d = belief_from_var[*e].take_or_clone(clear_incoming);
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
    clear_incoming: bool,
    pub_red: &PublicValue,
) -> impl Iterator<Item = Distribution> + 'a {
    // Special case for single-input XOR
    if factor.edges.len() == 2 {
        return dest
            .iter()
            .map(|var| {
                let i = factor.edges.get_index_of(var).unwrap();
                let mut distr = belief_from_var[factor.edges[1 - i]].take_or_clone(clear_incoming);
                distr.xor_cst(pub_red);
                distr
            })
            .collect::<Vec<_>>()
            .into_iter();
    }
    let mut acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
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
        .filter(|(e, _)| !belief_from_var[**e].is_full());
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
                    let mut d = belief_from_var[*e].take_or_clone(clear_incoming);
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
            let mut d = belief_from_var[*e].take_or_clone(clear_incoming);
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
    belief_from_var: &'a mut EdgeSlice<Distribution>,
    dest: &'a [VarId],
    clear_incoming: bool,
    pub_reduced: &PublicValue,
    inv_cst: u32,
) -> impl Iterator<Item = Distribution> + 'a {
    factor_xor(
        factor,
        belief_from_var,
        dest,
        clear_incoming,
        &pub_reduced.map(|x| x ^ inv_cst),
    )
}

fn factor_add<'a>(
    factor: &'a Factor,
    belief_from_var: &'a mut EdgeSlice<Distribution>,
    dest: &'a [VarId],
    clear_incoming: bool,
    pub_red: &PublicValue,
    plans: &FftPlans,
) -> impl Iterator<Item = Distribution> + 'a {
    let FactorKind::Assign {
        expr: ExprFactor::ADD { vars_neg },
        has_res,
    } = &factor.kind
    else {
        unreachable!()
    };
    // Special case for single-input ADD
    if factor.edges.len() == 2 {
        return dest
            .iter()
            .map(|dest_var| {
                let i_var = factor.edges.get_index_of(dest_var).unwrap();
                let i_op = 1 - i_var;
                let mut distr = belief_from_var[factor.edges[i_op]].take_or_clone(clear_incoming);
                if vars_neg[i_var] ^ vars_neg[i_op] ^ !has_res {
                    distr.negate();
                }
                distr.add_cst(pub_red, i_var != 0);
                distr
            })
            .collect::<Vec<_>>()
            .into_iter();
    }

    let mut taken_dest = vec![false; factor.edges.len()];
    let mut negated_vars = vars_neg.clone();
    if *has_res {
        negated_vars[0] = !negated_vars[0];
    }
    for dest in dest {
        taken_dest[factor.edges.get_index_of(dest).unwrap()] = true;
    }
    let mut uniform_iter = factor
        .edges
        .iter()
        .zip(taken_dest.iter())
        .zip(negated_vars.iter())
        .filter(|(((_, e), _), _)| !belief_from_var[**e].is_full());
    let uniform_op = uniform_iter.next();
    let uniform_template = belief_from_var[factor.edges[0]].as_uniform();
    let (nmulti, nc) = uniform_template.shape();
    let mut fft_tmp = ndarray::Array2::zeros((nmulti, nc / 2 + 1));
    let mut acc_fft = ndarray::Array2::zeros((nmulti, nc / 2 + 1));
    let mut acc_fft_init = false;
    let mut fft_scratch = plans.r2c.make_scratch_vec();
    let mut fft_input_scratch = plans.r2c.make_input_vec();
    if !pub_red.is_zero() {
        let acc = belief_from_var[factor.edges[0]].new_constant(pub_red);
        acc.fft_to(
            fft_input_scratch.as_mut_slice(),
            acc_fft.view_mut(),
            fft_scratch.as_mut_slice(),
            plans,
            false,
        );
        acc_fft_init = true;
    }
    if let Some((((v_dest, e_dest), t), dest_negated)) = uniform_op {
        if !*t || uniform_iter.next().is_some() {
            // At least 2 uniform operands, or single uniform is not in dest,
            // all dest messages are uniform.
            reset_incoming(factor, belief_from_var, &taken_dest, clear_incoming);
            return vec![uniform_template; dest.len()].into_iter();
        }
        // Single uniform op, only compute for that one.
        for (e, negated_var) in factor.edges.values().zip(negated_vars.iter()) {
            if e != e_dest {
                let negate = !(dest_negated ^ negated_var);
                belief_from_var[*e].fft_to(
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
                belief_from_var[*e].reset();
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
                belief_from_var[*e].fft_to(
                    fft_input_scratch.as_mut_slice(),
                    fft_e.view_mut(),
                    fft_scratch.as_mut_slice(),
                    plans,
                    *negated_var,
                );

                dest_fft.push(fft_e);
            } else {
                belief_from_var[*e].fft_to(
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
                belief_from_var[*e].reset();
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
    belief_from_var: &'a mut EdgeSlice<Distribution>,
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
                            res.op_multiply(&belief_from_var[*e])
                        } else {
                            res.op_multiply_factor(&belief_from_var[*e])
                        }
                    } else {
                        belief_from_var[*e].take_or_clone(clear_incoming)
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
                    belief_from_var[*e].reset();
                }
            }
            None
        }
    })
}

fn factor_lookup<'a>(
    factor: &'a Factor,
    belief_from_var: &'a mut EdgeSlice<Distribution>,
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
            distr.map_table(table.values.as_slice())
        } else {
            distr.map_table_inv(table.values.as_slice())
        };
        if clear_incoming {
            belief_from_var[factor.edges[1 - i]].reset();
        }
        res
    })
}

fn factor_gen_factor<'a>(
    factor: &'a Factor,
    belief_from_var: &'a mut EdgeSlice<Distribution>,
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
    let nmulti_actual = if factor.multi { nmulti } else { 1 };

    let res: Vec<Distribution> = dest
        .iter()
        .map(|dest| {
            let dest_idx = factor.edges.get_index_of(dest).unwrap();
            let op_dest_idx = operands
                .iter()
                .position(|op| {
                    if let GenFactorOperand::Var { factor_edge_id, .. } = op {
                        *factor_edge_id == dest_idx
                    } else {
                        false
                    }
                })
                .expect("must have dest operand");
            let mut distr = belief_from_var[factor.edges[dest_idx]].clone();
            distr.ensure_full();
            match gen_factor {
                GenFactor::Single(gen_factor) => {
                    factor_gen_factor_range_distr(
                        factor,
                        belief_from_var,
                        public_values,
                        nc,
                        operands,
                        dest_idx,
                        op_dest_idx,
                        &mut distr,
                        0..nmulti_actual,
                        gen_factor,
                    );
                }
                GenFactor::Multi(gfs) => {
                    assert_eq!(gfs.len(), nmulti_actual);
                    for (i, gen_factor) in gfs.iter().enumerate() {
                        factor_gen_factor_range_distr(
                            factor,
                            belief_from_var,
                            public_values,
                            nc,
                            operands,
                            dest_idx,
                            op_dest_idx,
                            &mut distr,
                            i..(i + 1),
                            gen_factor,
                        );
                    }
                }
            };
            distr
        })
        .collect();
    if clear_incoming {
        for e in factor.edges.values() {
            belief_from_var[*e].reset();
        }
    }
    res.into_iter()
}

fn factor_gen_factor_range_distr(
    factor: &Factor,
    belief_from_var: &mut EdgeSlice<Distribution>,
    public_values: &[PublicValue],
    nc: usize,
    operands: &[GenFactorOperand],
    dest_idx: usize,
    op_dest_idx: usize,
    distr: &mut Distribution,
    idxs: std::ops::Range<usize>,
    gen_factor: &GenFactorInner,
) {
    match gen_factor {
        GenFactorInner::Dense(gen_factor) => {
            for i in idxs {
                factor_gen_factor_dense_dest(
                    factor,
                    belief_from_var,
                    public_values,
                    nc,
                    operands,
                    dest_idx,
                    op_dest_idx,
                    distr,
                    i,
                    gen_factor,
                );
            }
        }
        GenFactorInner::SparseFunctional(gen_factor) => {
            factor_gen_factor_sparse_dest(
                factor,
                belief_from_var,
                public_values,
                nc,
                operands,
                dest_idx,
                op_dest_idx,
                distr,
                idxs,
                gen_factor,
            );
        }
    }
}

fn factor_gen_factor_dense_dest(
    factor: &Factor,
    belief_from_var: &mut EdgeSlice<Distribution>,
    public_values: &[PublicValue],
    nc: usize,
    operands: &[GenFactorOperand],
    dest_idx: usize,
    op_dest_idx: usize,
    distr: &mut Distribution,
    i: usize,
    gen_factor: &ndarray::ArrayD<f64>,
) {
    assert_eq!(gen_factor.shape().len(), operands.len());
    // First slice the array with the constants.
    let gen_factor = gen_factor.slice_each_axis(|ax| match operands[ax.axis.index()] {
        fg::GenFactorOperand::Var { .. } => ndarray::Slice::new(0, None, 1),
        fg::GenFactorOperand::Pub { pub_id } => {
            let mut pub_val = public_values[factor.publics[pub_id].0].get(i) as isize;
            if factor.publics[pub_id].1 {
                if nc.is_power_of_two() {
                    pub_val = !pub_val;
                } else {
                    // TODO Check that we enforce this at graph creation time and return a proper error.
                    panic!("Cannot negate operands with non-power-of-two number of classes.");
                }
            }
            ndarray::Slice::new(pub_val, Some(pub_val + 1), 1)
        }
    });
    let mut gen_factor = gen_factor.to_owned();
    for (op_idx, op) in operands.iter().enumerate() {
        if let fg::GenFactorOperand::Var {
            factor_edge_id,
            negated,
        } = op
        {
            if *factor_edge_id != dest_idx {
                if *negated {
                    todo!("Negated operands on generalized factors not yet implemented.");
                }
                let distr = &belief_from_var[factor.edges[*factor_edge_id]];
                let mut new_gen_factor: ndarray::ArrayD<f64> = ndarray::ArrayD::zeros(
                    gen_factor
                        .slice_axis(ndarray::Axis(op_idx), ndarray::Slice::new(0, Some(1), 1))
                        .shape(),
                );
                if let Some(distr) = distr.value() {
                    for (d, gf) in distr
                        .slice(s![i, ..])
                        .iter()
                        .zip(gen_factor.axis_chunks_iter(ndarray::Axis(op_idx), 1))
                    {
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
    for _ in 0..op_dest_idx {
        gen_factor.index_axis_inplace(ndarray::Axis(0), 0);
    }
    for _ in (op_dest_idx + 1)..operands.len() {
        gen_factor.index_axis_inplace(ndarray::Axis(1), 0);
    }
    distr
        .value_mut()
        .unwrap()
        .slice_mut(s![i, ..])
        .assign(&gen_factor);
}

fn gen_factor_operands_idxs(
    operands: &[GenFactorOperand],
    dest_idx: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut acc_ops_var_idx: Vec<usize> = vec![];
    let mut acc_ops_varpub_idx: Vec<usize> = vec![];
    let mut acc_ops_pub_idx: Vec<usize> = vec![];
    for (op_idx, op) in operands.iter().enumerate() {
        match op {
            fg::GenFactorOperand::Var { factor_edge_id, .. } => {
                if *factor_edge_id != dest_idx {
                    acc_ops_var_idx.push(op_idx);
                }
            }
            fg::GenFactorOperand::Pub { pub_id } => {
                acc_ops_pub_idx.push(*pub_id);
                acc_ops_varpub_idx.push(op_idx);
            }
        }
    }
    return (acc_ops_var_idx, acc_ops_varpub_idx, acc_ops_pub_idx);
}

fn factor_gen_factor_sparse_dest(
    factor: &Factor,
    belief_from_var: &mut EdgeSlice<Distribution>,
    public_values: &[PublicValue],
    nc: usize,
    operands: &[GenFactorOperand],
    dest_idx: usize,
    op_dest_idx: usize,
    distr: &mut Distribution,
    idxs: std::ops::Range<usize>,
    gen_factor: &ndarray::Array2<u32>,
) {
    assert_eq!(gen_factor.shape()[1], operands.len());
    let (acc_ops_var_idx, acc_ops_varpub_idx, acc_ops_pub_idx) =
        gen_factor_operands_idxs(operands, dest_idx);
    // Use half of L3 cache (per core) with distributions (8 bytes per entry).
    let chunk_size = std::cmp::max(1, crate::L3_CORE / 2 / 8 / distr.shape().1 / operands.len());
    for idxs in idxs.range_chunks(chunk_size) {
        // Copy operand distributions to nice memory order
        let ops_distr_chunk = operands
            .iter()
            .map(|operand| {
                match operand {
                    fg::GenFactorOperand::Var { factor_edge_id, .. } => {
                        // Uniform distributions can be ignored in the factor computation.
                        if *factor_edge_id != dest_idx {
                            let distr = &belief_from_var[factor.edges[*factor_edge_id]];
                            if let Some(distr) = distr.value() {
                                let mut distr_f =
                                    ndarray::Array::zeros((idxs.len(), nc as usize).f());
                                distr_f.assign(&distr.slice(s![idxs.clone(), ..]));
                                return Some(distr_f);
                            }
                        }
                        None
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        let mut dest_all = ndarray::Array::zeros((idxs.len(), nc as usize).f());
        gen_factor_sparse_inner(
            idxs.clone(),
            nc,
            op_dest_idx,
            factor,
            gen_factor,
            ops_distr_chunk,
            public_values,
            &mut dest_all,
            &acc_ops_var_idx,
            &acc_ops_varpub_idx,
            &acc_ops_pub_idx,
        );
        distr
            .value_mut()
            .unwrap()
            .slice_mut(s![idxs.clone(), ..])
            .assign(&dest_all);
    }
}

#[inline(never)]
fn gen_factor_sparse_inner<'a>(
    idxs: std::ops::Range<usize>,
    nc: usize,
    op_dest_idx: usize,
    factor: &'a Factor,
    gen_factor: &Array2<u32>,
    ops_distr_chunk: Vec<Option<Array2<f64>>>,
    public_values: &'a [PublicValue],
    dest_all: &mut Array2<f64>,
    acc_ops_var_idx: &[usize],
    acc_ops_varpub_idx: &[usize],
    acc_ops_pub_idx: &[usize],
) {
    let mut prod_chunk = ndarray::Array1::<f64>::ones(idxs.len());
    let has_pub: bool = acc_ops_pub_idx.len() > 0;
    for op_values in gen_factor.outer_iter() {
        prod_chunk.fill(1.0);
        for op_idx in acc_ops_var_idx.iter() {
            if let Some(distr) = &ops_distr_chunk[*op_idx] {
                prod_chunk.mul_assign(&distr.slice(s![.., op_values[*op_idx] as usize]));
            }
        }
        if has_pub {
            for (pub_id, op_idx) in acc_ops_pub_idx.iter().zip(acc_ops_varpub_idx) {
                let op_val = op_values[*op_idx];
                for i in idxs.clone() {
                    let mut pub_val = public_values[factor.publics[*pub_id].0].get(i);
                    if factor.publics[*pub_id].1 {
                        if nc.is_power_of_two() {
                            pub_val = !pub_val & ((nc - 1) as ClassVal);
                        } else {
                            // TODO Check that we enforce this at graph creation time and return a proper error.
                            panic!(
                                "Cannot negate operands with non-power-of-two number of classes."
                            );
                        }
                    }
                    if pub_val != op_val {
                        prod_chunk[i - idxs.start] = 0.0;
                    }
                }
            }
        }

        *&mut dest_all.slice_mut(s![.., op_values[op_dest_idx] as usize]) += &prod_chunk;
    }
}
