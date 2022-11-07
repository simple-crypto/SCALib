
use super::factor_graph as fg;
use super::fg_parser;
use super::{ClassVal, NamedList};
use indexmap::IndexMap;

#[derive(Debug, Clone)]
enum GraphBuildError {
    MultipleTableDecl(String),
    MultipleVarDecl(String),
    UnknownVar(String),
    UnknownTable(String),
    RepeatedOperand(String),
    TableSize(usize),
    MissingTableDef(String),
    MultipleNc,
    NoNc,
}

impl fg::FactorGraph {
    fn build(nc: usize) -> Self {
        Self {
            nc,
            vars: NamedList::new(),
            factors: Vec::new(),
            edges: Vec::new(),
            publics: NamedList::new(),
            tables: NamedList::new(),
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
        self.vars.insert(
            name,
            fg::Var {
                multi,
                edges: IndexMap::new(),
            },
        );
        Ok(())
    }
    fn add_pub(&mut self, name: String, multi: bool) -> Result<(), GraphBuildError> {
        self.check_new_var(&name)?;
        self.publics.insert(name, fg::Public { multi });
        Ok(())
    }
    fn add_table(
        &mut self,
        name: String,
        values: Vec<ClassVal>,
    ) -> Result<(), GraphBuildError> {
        if self.tables.contains_key(&name) {
            return Err(GraphBuildError::MultipleTableDecl(name));
        }
        if values.len() != self.nc {
            return Err(GraphBuildError::TableSize(values.len()));
        }
        self.tables.insert(name, fg::Table { values });
        Ok(())
    }
    fn add_factor<'a>(
        &mut self,
        kind: fg::FactorKind<&str>,
        vars: impl Iterator<Item = &'a str>,
    ) -> Result<(), GraphBuildError> {
        let factor_id = self.factors.len();
        let mut edges = IndexMap::new();
        let mut publics = Vec::new();
        let mut multi = false;
        for var in vars {
            if let Some((var_id, _, v)) = self.vars.get_full_mut(var) {
                let edge_id = self.edges.len();
                if edges.insert(var_id, edge_id).is_some() {
                    return Err(GraphBuildError::RepeatedOperand(var.to_owned()));
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
                publics.push(pub_id);
                multi |= public.multi;
            } else {
                return Err(GraphBuildError::UnknownVar(var.to_owned()));
            }
        }
        let kind = kind.map_table(|t| {
            self.tables
                .get_index_of(t)
                .ok_or_else(|| GraphBuildError::UnknownTable(t.to_owned()))
        })?;
        self.factors.push(fg::Factor {
            kind,
            multi,
            edges,
            publics,
        });
        Ok(())
    }
}
    impl<T> fg::FactorKind<T> {
        fn map_table<S, E, F>(self, f: F) -> Result<fg::FactorKind<S>, E>
        where
            F: Fn(T) -> Result<S, E>,
        {
            Ok(match self {
                fg::FactorKind::AND => fg::FactorKind::AND,
                fg::FactorKind::OR => fg::FactorKind::OR,
                fg::FactorKind::XOR => fg::FactorKind::XOR,
                fg::FactorKind::NOT => fg::FactorKind::NOT,
                fg::FactorKind::ADD => fg::FactorKind::ADD,
                fg::FactorKind::MUL => fg::FactorKind::MUL,
                fg::FactorKind::LOOKUP { table } => fg::FactorKind::LOOKUP { table: f(table)? },
            })
        }
    }
impl fg_parser::Expr {
    fn as_factor_kind(&self) -> fg::FactorKind<&str> {
        match self {
            Self::Not(_) => fg::FactorKind::NOT,
            Self::Lookup { table, .. } => fg::FactorKind::LOOKUP {
                table: table.as_str(),
            },
            Self::Add(_) => fg::FactorKind::ADD,
            Self::Mul(_) => fg::FactorKind::MUL,
            Self::Xor(_) => fg::FactorKind::XOR,
            Self::And(_) => fg::FactorKind::AND,
            Self::Or(_) => fg::FactorKind::OR,
        }
    }
    fn vars(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::Not(var) | Self::Lookup { var, .. } => vec![var.0.as_str()],
            Self::Add(v) | Self::Mul(v) => v.iter().map(|v| v.0.as_str()).collect(),
            Self::Xor(v) | Self::And(v) | Self::Or(v) => v
                .iter()
                .map(|v| if v.neg { todo!() } else { v.var.0.as_str() })
                .collect(),
        }
        .into_iter()
    }
}
pub(super) fn build_graph(
    stmts: &[fg_parser::Statement],
    mut tables: NamedList<Vec<ClassVal>>,
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
            | fg_parser::Statement::End
            | fg_parser::Statement::Invalid
            | fg_parser::Statement::Comment
            | fg_parser::Statement::Property { .. }
            | fg_parser::Statement::NC(_) => {}
        }
    }
    for s in stmts {
        if let fg_parser::Statement::Property { dest, expr } = s {
            graph.add_factor(
                expr.as_factor_kind(),
                std::iter::once(dest.0.as_str()).chain(expr.vars()),
            )?;
        }
    }
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
