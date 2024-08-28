use std::{fmt, usize};

use super::{ClassVal, NamedList};
use indexmap::IndexMap;
use thiserror::Error;

macro_rules! new_id {
    ($it:ident, $vt:ident) => {
        index_vec::define_index_type! {
            pub struct $it = u32;
            DISPLAY_FORMAT = "{}";
        }
        pub type $vt<T> = index_vec::IndexVec<$it, T>;
        impl $it {
            pub fn idx(self) -> usize {
                <Self as Into<usize>>::into(self)
            }
            pub fn from_idx(x: usize) -> Self {
                <Self as From<usize>>::from(x)
            }
        }
    };
}
new_id!(VarId, VarVec);
new_id!(FactorId, FactorVec);
new_id!(EdgeId, EdgeVec);
pub type EdgeSlice<T> = index_vec::IndexSlice<EdgeId, [T]>;
pub type PublicId = usize;
pub type TableId = usize;
pub type GenFactorId = usize;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Var {
    pub(super) multi: bool,
    pub(super) edges: IndexMap<FactorId, EdgeId>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Factor {
    pub(super) kind: FactorKind,
    pub(super) multi: bool,
    // res is first element (if there is a res), operands come next
    pub(super) edges: IndexMap<VarId, EdgeId>,
    // May not be allowed for all factor kinds
    pub(super) publics: Vec<(PublicId, bool)>,
}

impl Factor {
    pub(super) fn var_id(&self, var_order: usize) -> VarId {
        *self.edges.get_index(var_order).unwrap().0
    }
    pub(super) fn res_id(&self) -> Option<VarId> {
        if let FactorKind::Assign { has_res, .. } = &self.kind {
            has_res.then_some(self.var_id(0))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum ExprFactor {
    AND { vars_neg: Vec<bool> },
    XOR,
    NOT,
    ADD { vars_neg: Vec<bool> },
    MUL,
    LOOKUP { table: TableId },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum GenFactorOperand {
    Var {
        factor_edge_id: usize,
        negated: bool,
    },
    Pub {
        pub_id: usize,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum FactorKind {
    Assign {
        expr: ExprFactor,
        // Is the result a variable (and not a public) ?
        has_res: bool,
    },
    GenFactor {
        id: GenFactorId,
        operands: Vec<GenFactorOperand>,
    },
}

impl ExprFactor {
    fn merge(&self, a: ClassVal, b: ClassVal, nc: usize, negb: bool) -> ClassVal {
        let (notb, minb) = if negb {
            (((nc - 1) as ClassVal) ^ b, (nc - (b as usize)) as ClassVal)
        } else {
            (b, b)
        };
        match self {
            Self::AND { vars_neg: _ } => a & notb,
            Self::XOR => a ^ notb,
            Self::ADD { vars_neg: _ } => (((a as u64) + (minb as u64)) % (nc as u64)) as ClassVal,
            Self::MUL => (((a as u64) * (minb as u64)) % (nc as u64)) as ClassVal,
            Self::NOT | Self::LOOKUP { .. } => unreachable!(),
        }
    }
    fn neutral(&self, nc: usize) -> ClassVal {
        match self {
            Self::AND { vars_neg: _ } => (nc - 1) as ClassVal,
            Self::XOR | Self::ADD { vars_neg: _ } => 0,
            Self::MUL => 1,
            Self::NOT | Self::LOOKUP { .. } => unreachable!(),
        }
    }
    fn neg_res(&self) -> bool {
        match self {
            Self::ADD { .. } => true,
            Self::XOR => false,
            Self::AND { .. } | Self::MUL | Self::NOT | Self::LOOKUP { .. } => {
                unimplemented!("Case to analyze and implement.")
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Edge {
    pub(super) var: VarId,
    pub(super) pos_var: usize,
    pub(super) factor: FactorId,
    pub(super) pos_factor: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Public {
    pub(super) multi: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct GenFactor {
    pub(super) multi: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Table {
    pub(super) values: Vec<ClassVal>,
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum Node {
    Var(VarId),
    Factor(FactorId),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FactorGraph {
    pub(super) nc: usize,
    pub(super) vars: NamedList<Var>,
    pub(super) factors: NamedList<Factor>,
    pub(super) edges: EdgeVec<Edge>,
    pub(super) publics: NamedList<Public>,
    pub(super) tables: NamedList<Table>,
    pub(super) gen_factors: NamedList<GenFactor>,
    pub(super) petgraph: petgraph::Graph<Node, EdgeId, petgraph::Undirected>,
    pub(super) var_graph_ids: VarVec<petgraph::graph::NodeIndex>,
    pub(super) factor_graph_ids: FactorVec<petgraph::graph::NodeIndex>,
    pub(super) cyclic_single: bool,
    pub(super) cyclic_multi: bool,
}

#[derive(Debug, Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
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
    pub fn iter(&self, n: usize) -> impl Iterator<Item = ClassVal> + '_ {
        if let PublicValue::Multi(x) = self {
            assert_eq!(x.len(), n);
        }
        (0..n).map(|i| self.get(i))
    }
    pub fn get(&self, n: usize) -> ClassVal {
        match self {
            PublicValue::Single(x) => *x,
            PublicValue::Multi(x) => x[n],
        }
    }
    pub fn map(&self, f: impl Fn(ClassVal) -> ClassVal) -> Self {
        match self {
            PublicValue::Single(x) => PublicValue::Single(f(*x)),
            PublicValue::Multi(x) => PublicValue::Multi(x.iter().cloned().map(f).collect()),
        }
    }
    pub fn is_zero(&self) -> bool {
        match self {
            PublicValue::Single(x) => *x == 0,
            PublicValue::Multi(x) => x.iter().all(|x| *x == 0),
        }
    }
}
impl fmt::Display for PublicValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PublicValue::Single(x) => write!(f, "{}", x),
            PublicValue::Multi(x) => write!(f, "{:?}", x.as_slice()),
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum FGError {
    #[error("No variable named {0}.")]
    NoVar(String),
    #[error("No factor named {0}.")]
    NoFactor(String),
    #[error("No edge between variable {var} and factor {factor}.")]
    NoEdge { var: String, factor: FactorId },
    #[error("Failure at factor {0}. Expected result {1}, got {2}.")]
    CheckFail(String, PublicValue, PublicValue),
    #[error("")]
    InvalidGenericFactorAssignment(String),
}

type FGResult<T> = Result<T, FGError>;

impl FactorGraph {
    pub fn edge(&self, var: VarId, factor: FactorId) -> FGResult<EdgeId> {
        self.var(var)
            .edges
            .get(&factor)
            .map(|e| *e)
            .ok_or_else(|| FGError::NoEdge {
                var: self.vars.get_index(var.idx()).unwrap().0.to_owned(),
                factor,
            })
    }
    pub fn public_multi(&self) -> impl Iterator<Item = (&str, bool)> {
        self.publics.iter().map(|(n, v)| (n.as_str(), v.multi))
    }
    pub fn gf_multi(&self) -> impl Iterator<Item = (&str, bool)> {
        self.gen_factors.iter().map(|(n, v)| (n.as_str(), v.multi))
    }
    pub fn get_varid(&self, var: &str) -> FGResult<VarId> {
        self.vars
            .get_index_of(var)
            .map(VarId::from_idx)
            .ok_or_else(|| FGError::NoVar(var.to_owned()))
    }
    pub(super) fn var(&self, var: VarId) -> &Var {
        &self.vars[var.idx()]
    }
    pub fn var_multi(&self, var: VarId) -> bool {
        self.var(var).multi
    }
    pub fn factor_multi(&self, factor: FactorId) -> bool {
        self.factor(factor).multi
    }
    pub fn edge_multi(&self, edge: EdgeId) -> bool {
        self.factor_multi(self.edges[edge].factor)
    }
    pub fn range_vars(&self) -> impl Iterator<Item = VarId> {
        (0..self.vars.len()).map(VarId::from_idx)
    }
    pub fn range_factors(&self) -> impl Iterator<Item = FactorId> {
        (0..self.factors.len()).map(FactorId::from_idx)
    }
    pub(super) fn factor(&self, factor: FactorId) -> &Factor {
        &self.factors[factor.idx()]
    }
    pub fn get_factorid(&self, factor: &str) -> FGResult<FactorId> {
        self.factors
            .get_index_of(factor)
            .map(FactorId::from_idx)
            .ok_or_else(|| FGError::NoFactor(factor.to_owned()))
    }
    pub fn var_names(&self) -> impl Iterator<Item = &str> {
        self.vars.keys().map(String::as_str)
    }
    pub fn vars(&self) -> impl Iterator<Item = (VarId, &str)> {
        self.vars
            .keys()
            .enumerate()
            .map(|(i, vn)| (VarId::from_idx(i), vn.as_str()))
    }
    pub fn var_name(&self, v: VarId) -> &str {
        self.vars.get_index(v.idx()).unwrap().0.as_str()
    }
    pub fn factor_names(&self) -> impl Iterator<Item = &str> {
        self.factors.keys().map(String::as_str)
    }
    pub fn factor_name(&self, f: FactorId) -> &str {
        self.factors.get_index(f.idx()).unwrap().0.as_str()
    }
    pub fn factor_scope<'s>(&'s self, factor: FactorId) -> impl Iterator<Item = VarId> + 's {
        self.factor(factor).edges.keys().cloned()
    }
    pub fn sanity_check(
        &self,
        public_values: Vec<PublicValue>,
        var_assignments: VarVec<PublicValue>,
        gen_factors: Vec<super::GenFactor>,
    ) -> FGResult<()> {
        assert_eq!(public_values.len(), self.publics.len());
        assert_eq!(var_assignments.len(), self.vars.len());
        let reduced_pub = self.reduce_pub(public_values.as_slice());
        for ((factor_name, factor), cst) in self.factors.iter().zip(reduced_pub) {
            match &factor.kind {
                FactorKind::Assign { expr, has_res } => {
                    let expected_res = factor
                        .res_id()
                        .map(|v_id| &var_assignments[v_id])
                        .unwrap_or(&cst);
                    let skip_res = if *has_res { 1 } else { 0 };
                    let mut ops = factor
                        .edges
                        .keys()
                        .skip(skip_res)
                        .map(|v_id| &var_assignments[*v_id]);
                    let res = match expr {
                        ExprFactor::AND { vars_neg } => {
                            let x = self.merge_pubs(
                                expr,
                                false,
                                ops.zip(vars_neg.iter().cloned())
                                    .chain(std::iter::once((&cst, false))),
                            );
                            // invert if we are doing an OR
                            match x {
                                PublicValue::Single(cv) if vars_neg[0] => {
                                    PublicValue::Single(self.not(cv))
                                }
                                PublicValue::Multi(cv) if vars_neg[0] => {
                                    PublicValue::Multi(cv.iter().map(|v| self.not(*v)).collect())
                                }
                                _ => x,
                            }
                        }
                        ExprFactor::XOR | ExprFactor::ADD { .. } | ExprFactor::MUL => self
                            .merge_pubs(
                                expr,
                                false,
                                ops.zip(std::iter::repeat(false))
                                    .chain(std::iter::once((&cst, false))),
                            ),
                        ExprFactor::NOT => ops.next().unwrap().map(|x| self.not(x)),
                        ExprFactor::LOOKUP { table } => ops
                            .next()
                            .unwrap()
                            .map(|x| self.tables[*table].values[x as usize]),
                    };
                    let check = match (&res, expected_res) {
                        (PublicValue::Single(v1), PublicValue::Multi(v2)) => {
                            v2.iter().any(|x| x != v1)
                        }
                        (PublicValue::Multi(v1), PublicValue::Single(v2)) => {
                            v1.iter().any(|x| x != v2)
                        }
                        (_, _) => &res != expected_res,
                    };
                    if check {
                        return Err(FGError::CheckFail(
                            factor_name.clone(),
                            expected_res.clone(),
                            res,
                        ));
                    }
                }
                FactorKind::GenFactor { id, operands } => {
                    let ops: Vec<&PublicValue> = operands
                        .iter()
                        .map(|op| match op {
                            GenFactorOperand::Var { factor_edge_id, .. } => {
                                &var_assignments[*factor_edge_id]
                            }
                            GenFactorOperand::Pub { pub_id } => &public_values[*pub_id],
                        })
                        .collect();
                    let nmulti_ops = ops.iter().find_map(|op| {
                        if let PublicValue::Multi(x) = op {
                            Some(x.len())
                        } else {
                            None
                        }
                    });
                    let nmulti_factor = if let super::GenFactor::Multi(gfv) = &gen_factors[*id] {
                        Some(gfv.len())
                    } else {
                        None
                    };
                    let nmulti = match (nmulti_ops, nmulti_factor) {
                        (Some(nm_ops), Some(nm_factors)) if nm_ops == nm_factors => nm_factors,
                        (Some(nm_ops), Some(nm_factors)) => {
                            return Err(FGError::InvalidGenericFactorAssignment(format!("Mismatch between multi declaration of GenFactor operands {} and GenFactor {}", nm_ops, nm_factors)))
                        }
                        (Some(nmulti), None) | (None, Some(nmulti)) => nmulti,
                        (None, None) => 1,
                    };
                    for i in 0..nmulti {
                        let gf = match &gen_factors[*id] {
                            super::GenFactor::Single(gfs) => gfs,
                            super::GenFactor::Multi(gfv) => &gfv[i],
                        };
                        let indices: Vec<ClassVal> = ops
                            .iter()
                            .map(|pv| match *pv {
                                PublicValue::Single(x) => *x,
                                PublicValue::Multi(xvec) => xvec[i],
                            })
                            .collect();
                        match gf {
                            super::GenFactorInner::Dense(dense_factor) => {
                                if !(dense_factor[indices
                                    .iter()
                                    .map(|x| *x as usize)
                                    .collect::<Vec<usize>>()
                                    .as_slice()]
                                    > 0.0)
                                {
                                    return Err(FGError::InvalidGenericFactorAssignment(format!(
                                        "Invalid assignment to {}: {:?}",
                                        factor_name.clone(),
                                        indices.clone()
                                    )));
                                }
                            }

                            super::GenFactorInner::SparseFunctional(sf_factor) => {
                                if !(sf_factor
                                    .outer_iter()
                                    .any(|x| x.as_slice().unwrap() == indices.as_slice()))
                                {
                                    return Err(FGError::InvalidGenericFactorAssignment(format!(
                                        "Invalid assignment to {}: {:?}",
                                        factor_name.clone(),
                                        indices.clone()
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn reduce_pub(&self, public_values: &[PublicValue]) -> FactorVec<PublicValue> {
        self.factors
            .values()
            .map(|factor| {
                let mut pubs = factor
                    .publics
                    .iter()
                    .map(|(pub_id, nv)| (&public_values[*pub_id], *nv));
                match &factor.kind {
                    // Not used
                    FactorKind::Assign {
                        expr: ExprFactor::LOOKUP { .. },
                        ..
                    }
                    | FactorKind::GenFactor { .. } => PublicValue::Single(0),
                    FactorKind::Assign {
                        expr: ExprFactor::NOT,
                        ..
                    } => pubs
                        .next()
                        .map(|(val, _)| val.clone())
                        .unwrap_or(PublicValue::Single(0)),
                    FactorKind::Assign { expr, has_res } => self.merge_pubs(expr, !has_res, pubs),
                }
            })
            .collect()
    }
    fn not(&self, x: ClassVal) -> ClassVal {
        ((self.nc - 1) as ClassVal) ^ x
    }
    fn merge_pubs<'a>(
        &self,
        expr: &ExprFactor,
        is_res_pub: bool,
        // bool is the "invserse" for every item
        mut pubs: impl Iterator<Item = (&'a PublicValue, bool)>,
    ) -> PublicValue {
        let merge_inner = |p1: PublicValue, (p2, nv2): (&PublicValue, bool)| {
            let f = |x: ClassVal, y: ClassVal| expr.merge(x, y, self.nc, nv2);
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
        };
        let neutral = PublicValue::Single(expr.neutral(self.nc));
        let init = if is_res_pub && expr.neg_res() {
            let (pub_res, pub_res_neg) = pubs.next().unwrap();
            merge_inner(neutral, (pub_res, !pub_res_neg))
        } else {
            neutral
        };
        pubs.fold(init, merge_inner)
    }

    pub(super) fn is_cyclic(&self, multi_exec: bool) -> bool {
        if multi_exec {
            self.cyclic_multi
        } else {
            self.cyclic_single
        }
    }

    pub(super) fn propagation_order(&self, var: VarId) -> Vec<(Node, Option<Node>)> {
        let mut propagations = vec![(Node::Var(var), None)];
        petgraph::visit::depth_first_search(&self.petgraph, [self.var_graph_ids[var]], |event| {
            if let petgraph::visit::DfsEvent::TreeEdge(parent, node) = event {
                propagations.push((self.petgraph[node], Some(self.petgraph[parent])));
            }
            petgraph::visit::Control::<()>::Continue
        });
        propagations.reverse();
        propagations
    }
}

impl Node {
    pub(super) fn var(self) -> Option<VarId> {
        if let Node::Var(id) = self {
            Some(id)
        } else {
            None
        }
    }
    pub(super) fn factor(self) -> Option<FactorId> {
        if let Node::Factor(id) = self {
            Some(id)
        } else {
            None
        }
    }
}
