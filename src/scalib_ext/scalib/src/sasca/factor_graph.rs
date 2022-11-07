use indexmap::IndexMap;



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

impl FactorGraph {
    pub fn edge(&self, var: VarId, factor: FactorId) -> Option<EdgeId> {
        self.vars[var].edges.get_index(factor).map(|(_, e)| *e)
    }
}

