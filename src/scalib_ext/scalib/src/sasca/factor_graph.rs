use std::fmt;

use indexmap::IndexMap;
use thiserror::Error;

use super::{ClassVal, NamedList};

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
    // Is the result a variable (and not a public) ?
    pub(super) has_res: bool,
    // May not be allowed for all factor kinds
    pub(super) publics: Vec<(PublicId, bool)>,
}

impl Factor {
    pub(super) fn var_id(&self, var_order: usize) -> VarId {
        *self.edges.get_index(var_order).unwrap().0
    }
    pub(super) fn res_id(&self) -> Option<VarId> {
        self.has_res.then_some(self.var_id(0))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum FactorKind<T = TableId> {
    AND { vars_neg: Vec<bool> },
    XOR,
    NOT,
    ADD,
    MUL,
    LOOKUP { table: T },
}

impl FactorKind {
    fn merge(&self, a: ClassVal, b: ClassVal, nc: usize) -> ClassVal {
        match self {
            FactorKind::AND { vars_neg: _ } => a & b,
            FactorKind::XOR => a ^ b,
            FactorKind::ADD => (((a as u64) + (b as u64)) % (nc as u64)) as ClassVal,
            FactorKind::MUL => (((a as u64) + (b as u64)) % (nc as u64)) as ClassVal,
            FactorKind::NOT | FactorKind::LOOKUP { .. } => unreachable!(),
        }
    }
    fn neutral(&self, nc: usize) -> ClassVal {
        match self {
            FactorKind::AND { vars_neg: _ } => (nc - 1) as ClassVal,
            FactorKind::XOR | FactorKind::ADD => 0,
            FactorKind::MUL => 1,
            FactorKind::NOT | FactorKind::LOOKUP { .. } => unreachable!(),
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
    pub(super) petgraph: petgraph::Graph<Node, EdgeId, petgraph::Undirected>,
    pub(super) var_graph_ids: VarVec<petgraph::graph::NodeIndex>,
    pub(super) factor_graph_ids: FactorVec<petgraph::graph::NodeIndex>,
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
    ) -> FGResult<()> {
        assert_eq!(public_values.len(), self.publics.len());
        assert_eq!(var_assignments.len(), self.vars.len());
        let reduced_pub = self.reduce_pub(public_values.as_slice());
        for ((factor_name, factor), cst) in self.factors.iter().zip(reduced_pub) {
            let expected_res = factor
                .res_id()
                .map(|v_id| &var_assignments[v_id])
                .unwrap_or(&cst);
            let skip_res = if factor.has_res { 1 } else { 0 };
            let mut ops = factor
                .edges
                .keys()
                .skip(skip_res)
                .map(|v_id| &var_assignments[*v_id]);
            let res = match &factor.kind {
                FactorKind::AND { vars_neg } => self.merge_pubs(
                    &factor.kind,
                    ops.zip(vars_neg.iter().cloned())
                        .chain(std::iter::once((&cst, false))),
                ),
                FactorKind::XOR | FactorKind::ADD | FactorKind::MUL => self.merge_pubs(
                    &factor.kind,
                    ops.zip(std::iter::repeat(false))
                        .chain(std::iter::once((&cst, false))),
                ),
                FactorKind::NOT => ops.next().unwrap().map(|x| self.not(x)),
                FactorKind::LOOKUP { table } => ops
                    .next()
                    .unwrap()
                    .map(|x| self.tables[*table].values[x as usize]),
            };
            if &res != expected_res {
                return Err(FGError::CheckFail(
                    factor_name.clone(),
                    expected_res.clone(),
                    res,
                ));
            }
        }
        Ok(())
    }
    pub(super) fn reduce_pub(&self, public_values: &[PublicValue]) -> FactorVec<PublicValue> {
        self.factors
            .values()
            .map(|factor| {
                match &factor.kind {
                    // Not used
                    FactorKind::NOT | FactorKind::LOOKUP { .. } => PublicValue::Single(0),
                    _ => self.merge_pubs(
                        &factor.kind,
                        factor
                            .publics
                            .iter()
                            .map(|(pub_id, nv)| (&public_values[*pub_id], *nv)),
                    ),
                }
            })
            .collect()
    }
    fn not(&self, x: ClassVal) -> ClassVal {
        ((self.nc - 1) as ClassVal) ^ x
    }
    fn merge_pubs<'a>(
        &self,
        factor_kind: &FactorKind,
        // bool is the "invserse" for every item
        pubs: impl Iterator<Item = (&'a PublicValue, bool)>,
    ) -> PublicValue {
        let merge_inner = |p1: PublicValue, (p2, nv2): (&PublicValue, bool)| {
            let f = |x: ClassVal, y: ClassVal| {
                factor_kind.merge(x, if nv2 { self.not(y) } else { y }, self.nc)
            };
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
        pubs.fold(
            PublicValue::Single(factor_kind.neutral(self.nc)),
            merge_inner,
        )
    }

    pub(super) fn is_cyclic(&self, multi_exec: bool) -> bool {
        if petgraph::algo::is_cyclic_undirected(&self.petgraph) {
            return true;
        }
        if multi_exec {
            return petgraph::algo::kosaraju_scc(&self.petgraph)
                .into_iter()
                .any(|scc| {
                    scc.into_iter()
                        .filter(|n| match self.petgraph[*n] {
                            Node::Var(var_id) => {
                                !self.vars.get_index(var_id.index()).unwrap().1.multi
                            }
                            Node::Factor(_) => false,
                        })
                        .count()
                        > 1
                });
        } else {
            return false;
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
