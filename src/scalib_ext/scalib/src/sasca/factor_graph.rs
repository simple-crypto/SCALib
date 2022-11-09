use indexmap::IndexMap;

use thiserror::Error;


use super::{ClassVal,NamedList};

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
    }
}
new_id!(VarId, VarVec);
new_id!(FactorId, FactorVec);
new_id!(EdgeId, EdgeVec);
pub type EdgeSlice<T> = index_vec::IndexSlice<EdgeId, [T]>;
pub type PublicId = usize;
pub type TableId = usize;

#[derive(Debug, Clone)]
pub(super) struct Var {
    pub(super) multi: bool,
    pub(super) edges: IndexMap<FactorId, EdgeId>,
}

#[derive(Debug, Clone)]
pub(super) struct Factor {
    pub(super) kind: FactorKind,
    pub(super) multi: bool,
    // res is first element, operands come next
    pub(super) edges: IndexMap<VarId, EdgeId>,
    // May not be allowed for all factor kinds
    pub(super) publics: Vec<PublicId>,
}

#[derive(Debug, Clone)]
pub(super) enum FactorKind<T = TableId> {
    AND { vars_neg: Vec<bool> },
    OR { vars_neg: Vec<bool> },
    XOR,
    NOT,
    ADD,
    MUL,
    LOOKUP { table: T },
}

#[derive(Debug, Clone)]
pub(super) struct Edge {
    pub(super) var: VarId,
    pub(super) pos_var: usize,
    pub(super) factor: FactorId,
    pub(super) pos_factor: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Public {
    pub(super) multi: bool,
}

#[derive(Debug, Clone)]
pub(super) struct Table {
    pub(super) values: Vec<ClassVal>,
}

pub struct FactorGraph {
    pub(super) nc: usize,
    pub(super) vars: NamedList<Var>,
    pub(super) factors: FactorVec<Factor>,
    pub(super) edges: EdgeVec<Edge>,
    pub(super) publics: NamedList<Public>,
    pub(super) tables: NamedList<Table>,
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

#[derive(Debug, Clone, Error)]
pub enum FGError {
    #[error("No variable named {0}.")]
    NoVar(String),
    #[error("No edge between variable {var} and factor {factor}.")]
    NoEdge {
        var: String,
        factor: FactorId,
    }
}

type Result<T> = std::result::Result<T, FGError>;

impl FactorGraph {
    pub fn edge(&self, var: VarId, factor: FactorId) -> Result<EdgeId> {
        self.var(var).edges.get(&factor).map(|e| *e).ok_or_else(|| FGError::NoEdge { var: self.vars.get_index(var.idx()).unwrap().0.to_owned(), factor })
    }
    pub fn public_multi(&self) -> impl Iterator<Item=(&str, bool)> {
        self.publics.iter().map(|(n, v)| (n.as_str(), v.multi))
    }
    pub fn get_varid(&self, var: &str) -> Result<VarId> {
        self.vars.get_index_of(var).map(VarId::from_idx).ok_or_else(|| FGError::NoVar(var.to_owned()))
    }
    pub(super) fn var(&self, var: VarId) -> &Var {
        &self.vars[var.idx()]
    }
    pub fn var_multi(&self, var: VarId) -> bool {
        self.var(var).multi
    }
    pub fn range_vars(&self) -> impl Iterator<Item=VarId> {
        (0..self.vars.len()).map(VarId::from_idx)
    }
    pub fn range_factors(&self) -> impl Iterator<Item=FactorId> {
        (0..self.factors.len()).map(FactorId::from_idx)
    }
}

