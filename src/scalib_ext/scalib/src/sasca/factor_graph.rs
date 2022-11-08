use indexmap::IndexMap;

use thiserror::Error;


use super::{ClassVal,NamedList};

pub(super) type VarId = usize;
pub(super) type FactorId = usize;
pub(super) type EdgeId = usize;
pub(super) type PublicId = usize;
pub(super) type TableId = usize;

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
    pub(super) factors: Vec<Factor>,
    pub(super) edges: Vec<Edge>,
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
        self.vars[var].edges.get_index(factor).map(|(_, e)| *e).ok_or_else(|| FGError::NoEdge { var: self.vars.get_index(var).unwrap().0.to_owned(), factor })
    }
    pub fn public_multi(&self) -> impl Iterator<Item=(&str, bool)> {
        self.publics.iter().map(|(n, v)| (n.as_str(), v.multi))
    }
    pub fn get_varid(&self, var: &str) -> Result<VarId> {
        self.vars.get_index_of(var).ok_or_else(|| FGError::NoVar(var.to_owned()))
    }
    pub fn var_multi(&self, var: VarId) -> bool {
        self.vars[var].multi
    }
}

