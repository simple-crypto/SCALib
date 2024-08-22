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
    #[error("Generic {0} declared multiple times.")]
    MultipleGenericDecl(String),
    #[error("Variable or public {0} declared multiple times.")]
    MultipleVarDecl(String),
    #[error("Multiple properties with name {0}, property name must be unique.")]
    MultiplePropDecl(String),
    #[error("Variable or public {0} not declared.")]
    UnknownVar(String),
    #[error("Table {0} not declared.")]
    UnknownTable(String),
    #[error("Generic factor {0} not declared.")]
    UnknownGenFactor(String),
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
            gen_factors: NamedList::new(),
            petgraph: petgraph::Graph::new_undirected(),
            var_graph_ids: VarVec::new(),
            factor_graph_ids: FactorVec::new(),
            cyclic_single: false,
            cyclic_multi: false,
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
    fn add_generic(&mut self, name: String, multi: bool) -> Result<(), GraphBuildError> {
        if self.gen_factors.contains_key(&name) {
            return Err(GraphBuildError::MultipleGenericDecl(name));
        }
        self.gen_factors.insert(name, fg::GenFactor { multi });
        Ok(())
    }
    fn add_factor<'a>(
        &mut self,
        name: String,
        vars: impl Iterator<Item = (&'a fg_parser::Var, bool)>,
        build_kind: impl FnOnce(&Self, &[bool]) -> Result<fg::FactorKind, GraphBuildError>,
        inv_pub_neg: bool,
    ) -> Result<(), GraphBuildError> {
        if self.factors.contains_key(&name) {
            return Err(GraphBuildError::MultiplePropDecl(name));
        }
        let factor_entry = self.factors.entry(name.clone());
        let factor_id = FactorId::from_idx(factor_entry.index());
        let mut edges = IndexMap::new();
        let mut publics = Vec::new();
        let mut multi = false;
        let mut is_pub = Vec::new();
        for (var, neg) in vars {
            let var = var.0.as_str();
            if let Some((var_id, _, v)) = self.vars.get_full_mut(var) {
                is_pub.push(false);
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
                is_pub.push(true);
                let neg = if inv_pub_neg { !neg } else { neg };
                publics.push((pub_id, neg));
                multi |= public.multi;
            } else {
                return Err(GraphBuildError::UnknownVar(var.to_owned()));
            }
        }
        let kind = build_kind(self, is_pub.as_slice())?;
        let factor = fg::Factor {
            kind,
            multi,
            edges,
            publics,
        };
        let factor_entry = self.factors.entry(name);
        factor_entry.or_insert(factor);
        self.factor_graph_ids
            .push(self.petgraph.add_node(Node::Factor(factor_id)));
        Ok(())
    }
    fn add_assign<'a>(
        &mut self,
        name: String,
        dest: &fg_parser::Var,
        expr: &fg_parser::Expr,
    ) -> Result<(), GraphBuildError> {
        let vars = std::iter::once((dest, expr.neg_res())).chain(expr.vars_neg().into_iter());
        self.add_factor(
            name,
            vars,
            |s, is_pub| {
                Ok(fg::FactorKind::Assign {
                    expr: expr.as_factor_expr(
                        |t| {
                            s.tables
                                .get_index_of(t)
                                .ok_or_else(|| GraphBuildError::UnknownTable(t.to_owned()))
                        },
                        is_pub[0],
                        &is_pub[1..],
                    )?,
                    has_res: !is_pub[0],
                })
            },
            matches!(expr, fg_parser::Expr::Or(_)),
        )
    }
    fn add_genfactor<'a>(
        &mut self,
        name: String,
        gen_factor: &str,
        vars: &[fg_parser::NVar],
    ) -> Result<(), GraphBuildError> {
        let var_list = vars.iter().map(|v| (&v.var, v.neg));
        self.add_factor(
            name,
            var_list,
            |s, is_pub| {
                let mut n_pubs = 0;
                let mut n_vars = 0;
                let mut operands = Vec::new();
                for (i, p) in is_pub.iter().enumerate() {
                    if *p {
                        operands.push(fg::GenFactorOperand::Pub(n_pubs));
                        n_pubs += 1;
                    } else {
                        operands.push(fg::GenFactorOperand::Var(n_vars, vars[i].neg));
                        n_vars += 1;
                    }
                }
                Ok(fg::FactorKind::GenFactor {
                    id: s
                        .gen_factors
                        .get_index_of(gen_factor)
                        .ok_or_else(|| GraphBuildError::UnknownGenFactor(gen_factor.to_owned()))?,
                    operands,
                })
            },
            false,
        )
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
    /// Return the "flattended" factor graph for 2 executions, which can then be used to check
    /// cyclicity for any nexec > 1.
    /// Needs correct .vars, .factors and .edges
    fn petgraph_2exec(
        &self,
    ) -> petgraph::Graph<(Node, u32), super::factor_graph::EdgeId, petgraph::Undirected> {
        enum Either<T> {
            Single(T),
            Multi([T; 2]),
        }
        let mut graph = petgraph::Graph::new_undirected();
        let var_graph_ids = self
            .vars
            .values()
            .enumerate()
            .map(|(var_id, var)| {
                if var.multi {
                    Either::Multi([
                        graph.add_node((Node::Var(VarId::from_usize(var_id)), 0)),
                        graph.add_node((Node::Var(VarId::from_usize(var_id)), 1)),
                    ])
                } else {
                    Either::Single(graph.add_node((Node::Var(VarId::from_usize(var_id)), 0)))
                }
            })
            .collect::<VarVec<_>>();
        let factor_graph_ids = self
            .factors
            .values()
            .enumerate()
            .map(|(factor_id, factor)| {
                if factor.multi {
                    Either::Multi([
                        graph.add_node((Node::Factor(FactorId::from_usize(factor_id)), 0)),
                        graph.add_node((Node::Factor(FactorId::from_usize(factor_id)), 1)),
                    ])
                } else {
                    Either::Single(
                        graph.add_node((Node::Factor(FactorId::from_usize(factor_id)), 0)),
                    )
                }
            })
            .collect::<FactorVec<_>>();
        for (i, e) in self.edges.iter_enumerated() {
            match (&var_graph_ids[e.var], &factor_graph_ids[e.factor]) {
                (Either::Single(var_node), Either::Single(factor_node)) => {
                    graph.add_edge(*var_node, *factor_node, i);
                }
                (Either::Multi(var_nodes), Either::Multi(factor_nodes)) => {
                    for (var_node, factor_node) in var_nodes.iter().zip(factor_nodes.iter()) {
                        graph.add_edge(*var_node, *factor_node, i);
                    }
                }
                (Either::Multi(nodes_multi), Either::Single(node_single))
                | (Either::Single(node_single), Either::Multi(nodes_multi)) => {
                    for node_multi in nodes_multi.iter() {
                        graph.add_edge(*node_single, *node_multi, i);
                    }
                }
            }
        }
        graph
    }
}
impl fg_parser::Expr {
    fn as_factor_expr<F>(
        &self,
        ft: F,
        res_public: bool,
        vars_public: &[bool],
    ) -> Result<fg::ExprFactor, GraphBuildError>
    where
        F: Fn(&str) -> Result<fg::TableId, GraphBuildError>,
    {
        let vars_neg = self
            .vars_neg()
            .into_iter()
            .zip(vars_public.iter())
            .filter(|(_, public)| !**public)
            .map(|((_, neg), _)| neg);
        let vars_neg: Vec<bool> = (!res_public)
            .then_some(self.neg_res())
            .into_iter()
            .chain(vars_neg)
            .collect();
        Ok(match self {
            Self::Not(_) => fg::ExprFactor::NOT,
            Self::Lookup { table, .. } => fg::ExprFactor::LOOKUP {
                table: ft(table.as_str())?,
            },
            Self::Sum(_) => fg::ExprFactor::ADD { vars_neg },
            Self::Mul(_) => fg::ExprFactor::MUL,
            Self::Xor(_) => fg::ExprFactor::XOR,
            Self::And(_) | Self::Or(_) => fg::ExprFactor::AND { vars_neg },
        })
    }
    /// Returns operands with their negation.
    /// Maps Or to And using De Morgan law.
    fn vars_neg(&self) -> Vec<(&fg_parser::Var, bool)> {
        match self {
            Self::Not(var) | Self::Lookup { var, .. } => vec![(var, false)],
            Self::Xor(v) | Self::Mul(v) => v.iter().map(|v| (v, false)).collect(),
            Self::Sum(v) => v
                .iter()
                .map(|signed_var| match signed_var.sign {
                    fg_parser::SumOperation::Add => (&signed_var.var, false),
                    fg_parser::SumOperation::Subtract => (&signed_var.var, true),
                })
                .collect(),
            Self::And(v) => v.iter().map(|v| (&v.var, v.neg)).collect(),
            Self::Or(v) => v.iter().map(|v| (&v.var, !v.neg)).collect(),
        }
    }
    /// Should we negate the result due to De Morgan law use ?
    fn neg_res(&self) -> bool {
        match self {
            Self::Or(_) => true,
            _ => false,
        }
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
            fg_parser::Statement::GenericDecl { name, multi } => {
                graph.add_generic(name.clone(), *multi)?;
            }
            fg_parser::Statement::Invalid
            | fg_parser::Statement::Empty
            | fg_parser::Statement::Property { .. }
            | fg_parser::Statement::NC(_) => {}
        }
    }
    let mut anon_names = (0..).map(|i| format!("ANONYMOUS_{}", i));
    for s in stmts {
        if let fg_parser::Statement::Property { name, prop } = s {
            let name = name.clone().unwrap_or_else(|| anon_names.next().unwrap());
            match prop {
                fg_parser::Property::Assign { dest, expr } => {
                    graph.add_assign(name, dest, expr)?;
                }
                fg_parser::Property::GenFactor { factor, vars } => {
                    graph.add_genfactor(name, factor, vars)?;
                }
            };
        }
    }
    graph.add_graph_edges();
    graph.cyclic_single = petgraph::algo::is_cyclic_undirected(&graph.petgraph);
    graph.cyclic_multi = petgraph::algo::is_cyclic_undirected(&graph.petgraph_2exec());
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
