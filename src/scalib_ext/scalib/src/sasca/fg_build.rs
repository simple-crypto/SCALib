use super::factor_graph as fg;
use super::factor_graph::{FactorId, FactorVec, Node, VarVec};
use super::fg_parser;
use super::{ClassVal, NamedList, VarId};
use indexmap::IndexMap;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum GraphBuildError {
    #[error("Table {0} declared multiple times.")]
    MultipleTableDecl(String),
    #[error("Variable or public {0} declared multiple times.")]
    MultipleVarDecl(String),
    #[error("Multiple properties with name {0}, property name must be unique.")]
    MultiplePropDecl(String),
    #[error("Variable or public {0} not declared.")]
    UnknownVar(String),
    #[error("Table {0} not declared.")]
    UnknownTable(String),
    #[error("Operand {1} appears multiple times in factor {0}.")]
    RepeatedOperand(String, String),
    #[error("Constants appears as both result and operand in factor {0}.")]
    CstOpRes(String),
    #[error("Wrong length {1} for table {0}.")]
    TableSize(String, usize),
    #[error("Wrong value {1} for table {0}.")]
    TableValue(String, ClassVal),
    #[error("Value of table {0} not given.")]
    MissingTableDef(String),
    #[error("NC given more than once.")]
    MultipleNc,
    #[error("NC not given.")]
    NoNc,
    #[error("Could not parse graph description.\n{0}")]
    Parse(String),
}

impl fg::FactorGraph {
    fn build(nc: usize) -> Self {
        Self {
            nc,
            vars: NamedList::new(),
            factors: NamedList::new(),
            edges: fg::EdgeVec::new(),
            publics: NamedList::new(),
            tables: NamedList::new(),
            petgraph: petgraph::Graph::new_undirected(),
            var_graph_ids: VarVec::new(),
            factor_graph_ids: FactorVec::new(),
        }
    }
    fn check_new_var(&self, name: &String) -> Result<(), GraphBuildError> {
        if self.publics.contains_key(name) || self.vars.contains_key(name) {
            return Err(GraphBuildError::MultipleVarDecl(name.clone()));
        }
        Ok(())
    }
    fn add_var(&mut self, name: String, multi: bool) -> Result<(), GraphBuildError> {
        self.check_new_var(&name)?;
        let (var_idx, _) = self.vars.insert_full(
            name,
            fg::Var {
                multi,
                edges: IndexMap::new(),
            },
        );
        self.var_graph_ids
            .push(self.petgraph.add_node(Node::Var(VarId::from_idx(var_idx))));
        Ok(())
    }
    fn add_pub(&mut self, name: String, multi: bool) -> Result<(), GraphBuildError> {
        self.check_new_var(&name)?;
        self.publics.insert(name, fg::Public { multi });
        Ok(())
    }
    fn add_table(&mut self, name: String, values: Vec<ClassVal>) -> Result<(), GraphBuildError> {
        if self.tables.contains_key(&name) {
            return Err(GraphBuildError::MultipleTableDecl(name));
        }
        if values.len() != self.nc {
            return Err(GraphBuildError::TableSize(name, values.len()));
        }
        for v in values.iter() {
            if (*v as usize) >= self.nc {
                return Err(GraphBuildError::TableValue(name, *v));
            }
        }
        self.tables.insert(name, fg::Table { values });
        Ok(())
    }
    fn add_factor<'a>(
        &mut self,
        name: String,
        kind: fg::FactorKind<&str>,
        vars: impl Iterator<Item = &'a str>,
    ) -> Result<(), GraphBuildError> {
        if self.factors.contains_key(&name) {
            return Err(GraphBuildError::MultiplePropDecl(name));
        }
        let factor_entry = self.factors.entry(name.clone());
        let factor_id = FactorId::from_idx(factor_entry.index());
        let mut edges = IndexMap::new();
        let mut publics = Vec::new();
        let mut multi = false;
        let mut has_res = None;
        let mut is_var = Vec::new();
        for (i, var) in vars.enumerate() {
            if let Some((var_id, _, v)) = self.vars.get_full_mut(var) {
                has_res = Some(has_res.unwrap_or(true));
                is_var.push(true);
                let var_id = fg::VarId::from_idx(var_id);
                let edge_id = fg::EdgeId::from_idx(self.edges.len());
                if edges.insert(var_id, edge_id).is_some() {
                    return Err(GraphBuildError::RepeatedOperand(name, var.to_owned()));
                }
                v.edges.insert(factor_id, edge_id);
                self.edges.push(fg::Edge {
                    var: var_id,
                    pos_var: v.edges.len() - 1,
                    factor: factor_id,
                    pos_factor: edges.len() - 1,
                });
                multi |= v.multi;
            } else if let Some((pub_id, _, public)) = self.publics.get_full(var) {
                if has_res == Some(false) {
                    return Err(GraphBuildError::CstOpRes(name));
                }
                has_res = Some(has_res.unwrap_or(false));
                is_var.push(false);
                publics.push((pub_id, kind.get_neg(i)));
                multi |= public.multi;
            } else {
                return Err(GraphBuildError::UnknownVar(var.to_owned()));
            }
        }
        let kind = kind
            .map_vars_neg(|vn| {
                vn.into_iter()
                    .zip(is_var.into_iter())
                    .filter(|(_, iv)| *iv)
                    .map(|(vn, _)| vn)
                    .collect()
            })
            .map_table(|t| {
                self.tables
                    .get_index_of(t)
                    .ok_or_else(|| GraphBuildError::UnknownTable(t.to_owned()))
            })?;
        let factor = fg::Factor {
            kind,
            multi,
            edges,
            has_res: has_res.expect("at least one var"),
            publics,
        };
        factor_entry.or_insert(factor);
        self.factor_graph_ids
            .push(self.petgraph.add_node(Node::Factor(factor_id)));
        Ok(())
    }
    fn add_graph_edges(&mut self) {
        for (i, e) in self.edges.iter_enumerated() {
            self.petgraph.add_edge(
                self.var_graph_ids[e.var],
                self.factor_graph_ids[e.factor],
                i,
            );
        }
    }
}
impl<T> fg::FactorKind<T> {
    fn map_table<S, E, F>(self, f: F) -> Result<fg::FactorKind<S>, E>
    where
        F: Fn(T) -> Result<S, E>,
    {
        Ok(match self {
            fg::FactorKind::AND { vars_neg } => fg::FactorKind::AND { vars_neg },
            fg::FactorKind::XOR => fg::FactorKind::XOR,
            fg::FactorKind::NOT => fg::FactorKind::NOT,
            fg::FactorKind::ADD => fg::FactorKind::ADD,
            fg::FactorKind::MUL => fg::FactorKind::MUL,
            fg::FactorKind::LOOKUP { table } => fg::FactorKind::LOOKUP { table: f(table)? },
        })
    }
    fn map_vars_neg<F>(self, f: F) -> Self
    where
        F: FnOnce(Vec<bool>) -> Vec<bool>,
    {
        match self {
            fg::FactorKind::AND { vars_neg } => fg::FactorKind::AND {
                vars_neg: f(vars_neg),
            },
            x => x,
        }
    }
    fn get_neg(&self, i: usize) -> bool {
        match self {
            fg::FactorKind::AND { vars_neg } => vars_neg[i],
            _ => false,
        }
    }
}
impl fg_parser::Expr {
    fn as_factor_kind(&self) -> fg::FactorKind<&str> {
        fn get_neg(vars: &Vec<fg_parser::NVar>, neg: bool) -> Vec<bool> {
            // Include the result of the operation
            std::iter::once(neg)
                .chain(vars.iter().map(|v| v.neg ^ neg))
                .collect()
        }
        match self {
            Self::Not(_) => fg::FactorKind::NOT,
            Self::Lookup { table, .. } => fg::FactorKind::LOOKUP {
                table: table.as_str(),
            },
            Self::Add(_) => fg::FactorKind::ADD,
            Self::Mul(_) => fg::FactorKind::MUL,
            Self::Xor(_) => fg::FactorKind::XOR,
            Self::And(vars) => fg::FactorKind::AND {
                vars_neg: get_neg(vars, false),
            },
            // Use De Morgan's law to convert OR to AND.
            Self::Or(vars) => fg::FactorKind::AND {
                vars_neg: get_neg(vars, true),
            },
        }
    }
    fn vars(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::Not(var) | Self::Lookup { var, .. } => vec![var.0.as_str()],
            Self::Xor(v) | Self::Add(v) | Self::Mul(v) => v.iter().map(|v| v.0.as_str()).collect(),
            Self::And(v) | Self::Or(v) => v.iter().map(|v| v.var.0.as_str()).collect(),
        }
        .into_iter()
    }
}
pub(super) fn build_graph(
    stmts: &[fg_parser::Statement],
    mut tables: std::collections::HashMap<String, Vec<ClassVal>>,
) -> Result<fg::FactorGraph, GraphBuildError> {
    let nc = get_nc(stmts)?;
    let mut graph = fg::FactorGraph::build(nc as usize);
    for s in stmts {
        match s {
            fg_parser::Statement::VarDecl(vd) => {
                graph.add_var(vd.name.0.clone(), vd.multi)?;
            }
            fg_parser::Statement::PubDecl(vd) => {
                graph.add_pub(vd.name.0.clone(), vd.multi)?;
            }
            fg_parser::Statement::TableDecl { name, val } => {
                let val = if let Some(val) = val {
                    val.clone()
                } else if let Some(val) = tables.get_mut(name) {
                    std::mem::replace(val, Vec::new())
                } else {
                    return Err(GraphBuildError::MissingTableDef(name.clone()));
                };
                graph.add_table(name.clone(), val)?;
            }
            fg_parser::Statement::Invalid
            | fg_parser::Statement::Empty
            | fg_parser::Statement::Property { .. }
            | fg_parser::Statement::NC(_) => {}
        }
    }
    let mut anon_names = (0..).map(|i| format!("ANONYMOUS_{}", i));
    for s in stmts {
        if let fg_parser::Statement::Property { name, dest, expr } = s {
            let name = name.clone().unwrap_or_else(|| anon_names.next().unwrap());
            graph.add_factor(
                name,
                expr.as_factor_kind(),
                std::iter::once(dest.0.as_str()).chain(expr.vars()),
            )?;
        }
    }
    graph.add_graph_edges();
    Ok(graph)
}

fn get_nc(stmts: &[fg_parser::Statement]) -> Result<u64, GraphBuildError> {
    let mut nc_decls = stmts.iter().filter_map(|s| {
        if let fg_parser::Statement::NC(nc) = s {
            Some(*nc)
        } else {
            None
        }
    });
    if let Some(nc) = nc_decls.next() {
        if nc_decls.next().is_some() {
            return Err(GraphBuildError::MultipleNc);
        } else {
            Ok(nc)
        }
    } else {
        return Err(GraphBuildError::NoNc);
    }
}
