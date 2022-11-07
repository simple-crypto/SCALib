use ariadne::{Color, Fmt, Label, Report, ReportKind, Source};
use chumsky::prelude::*;
use indexmap::IndexMap;

enum GraphError {
    TODO,
    MultipleVarDecl(Var),
    MultipleNc,
    NoNc,
    MultipleTableDecl(String),
    TableSize(u64, u64),
}

////

mod fg {
    use indexmap::IndexMap;
    type NamedList<T> = IndexMap<String, T>;

    type ClassVal = u32;

    type VarId = usize;
    type FactorId = usize;
    type EdgeId = usize;
    type PublicId = usize;
    type TableId = usize;

    #[derive(Debug, Clone)]
    struct Var {
        multi: bool,
        edges: IndexMap<FactorId, EdgeId>,
    }

    #[derive(Debug, Clone)]
    struct Factor {
        kind: FactorKind,
        multi: bool,
        // res is first element, operands come next
        edges: IndexMap<VarId, EdgeId>,
        // May not be allowed for all factor kinds
        publics: Vec<PublicId>,
    }

    #[derive(Debug, Clone)]
    enum FactorKind<T = TableId> {
        AND,
        OR,
        XOR,
        NOT,
        ADD,
        MUL,
        LOOKUP { table: T },
    }
    impl<T> FactorKind<T> {
        fn map_table<S, E, F>(self, f: F) -> Result<FactorKind<S>, E>
        where
            F: Fn(T) -> Result<S, E>,
        {
            Ok(match self {
                FactorKind::AND => FactorKind::AND,
                FactorKind::OR => FactorKind::OR,
                FactorKind::XOR => FactorKind::XOR,
                FactorKind::NOT => FactorKind::NOT,
                FactorKind::ADD => FactorKind::ADD,
                FactorKind::MUL => FactorKind::MUL,
                FactorKind::LOOKUP { table } => FactorKind::LOOKUP { table: f(table)? },
            })
        }
    }

    #[derive(Debug, Clone)]
    struct Edge {
        var: VarId,
        pos_var: usize,
        factor: FactorId,
        pos_factor: usize,
    }

    #[derive(Debug, Clone)]
    struct Public {
        multi: bool,
    }

    #[derive(Debug, Clone)]
    struct Table {
        values: Vec<ClassVal>,
    }

    pub struct FactorGraph {
        nc: usize,
        vars: NamedList<Var>,
        factors: Vec<Factor>,
        edges: Vec<Edge>,
        publics: NamedList<Public>,
        tables: NamedList<Table>,
    }

    impl FactorGraph {
        pub fn build(nc: usize) -> Self {
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
            if self.publics.contains_key(&name) || self.vars.contains_key(&name) {
                return Err(GraphBuildError::MultipleVarDecl(name));
            }
            Ok(())
        }
        pub fn add_var(&mut self, name: String, multi: bool) -> Result<(), GraphBuildError> {
            self.check_new_var(&name)?;
            self.vars.insert(
                name,
                Var {
                    multi,
                    edges: IndexMap::new(),
                },
            );
            Ok(())
        }
        pub fn add_pub(&mut self, name: String, multi: bool) -> Result<(), GraphBuildError> {
            self.check_new_var(&name)?;
            self.publics.insert(name, Public { multi });
            Ok(())
        }
        pub fn add_table(
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
            self.tables.insert(name, Table { values });
            Ok(())
        }
        pub fn add_factor<'a>(
            &mut self,
            kind: FactorKind<&str>,
            vars: impl Iterator<Item = &'a str>,
        ) -> Result<(), GraphBuildError> {
            let factor_id = self.factors.len();
            let mut edges = IndexMap::new();
            let mut publics = Vec::new();
            let mut multi = false;
            for var in vars {
                if let Some((var_id, _, v)) = self.vars.get_full(var) {
                    let edge_id = self.edges.len();
                    if edges.insert(var_id, edge_id).is_some() {
                        return Err(GraphBuildError::RepeatedOperand(var.to_owned()));
                    }
                    v.edges.insert(factor_id, edge_id);
                    self.edges.push(Edge {
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
            self.factors.push(Factor {
                kind,
                multi,
                edges,
                publics,
            });
            Ok(())
        }
    }
}

////

#[derive(Debug, Clone)]
struct Var(String);

#[derive(Debug, Clone)]
struct NVar {
    var: Var,
    neg: bool,
}

impl NVar {
    fn new(var: Var, neg: bool) -> Self {
        Self { var, neg }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Not(Var),
    Lookup { var: Var, table: String },
    Add(Vec<Var>),
    Mul(Vec<Var>),
    Xor(Vec<NVar>),
    And(Vec<NVar>),
}
impl Expr {
    fn as_factor_kind(&self) -> fg::FactorKind<&str> {
        match self {
            Expr::Not(_) => fg::FactorGraph::Not,
            Expr::Lookup { table, .. } => fg::FactorGraph::LOOKUP {
                table: table.as_str(),
            },
            Expr::Add(_) => fg::FactorGraph::ADD,
            Expr::Mul(_) => fg::FactorGraph::MUL,
            Expr::Xor(_) => fg::FactorGraph::XOR,
            Expr::And(_) => fg::FactorGraph::AND,
        }
    }
    fn vars(&self) -> impl Iterator<Item = &str> {
        match self {
            Expr::Not(var) | Expr::Lookup { var, .. } => vec![var.0.as_str()],
            Expr::Add(v) | Expr::Mul(v) => v.iter().map(|v| v.0.as_str()).collect(),
            Expr::Xor(v) | Expr::And(v) => v
                .iter()
                .map(|v| if v.neg { todo!() } else { v.var.as_str() })
                .collect(),
        }
        .into_iter()
    }
}

type ClassVal = u32;

#[derive(Debug, Clone)]
struct VarDecl {
    name: Var,
    multi: bool,
}

#[derive(Debug, Clone)]
enum Statement {
    End, // only to appease parser
    Invalid,
    Comment,
    Property {
        dest: Var,
        expr: Expr,
    },
    NC(u64),
    VarDecl(VarDecl),
    PubDecl(VarDecl),
    TableDecl {
        name: String,
        val: Option<Vec<ClassVal>>,
    },
}

fn parser() -> impl Parser<char, Vec<Statement>, Error = Simple<char>> {
    let pad = just(' ').or(just('\t')).repeated();
    let space = pad.at_least(1);
    let op = |c| just(c).delimited_by(pad, pad);
    let ident = pad.ignore_then(text::ident()).then_ignore(pad);
    let kw = |s| text::keyword::<_, _, Simple<char>>(s).delimited_by(pad, space);
    let var = ident.map(|s| Var(s));
    let not_var = || op('!').ignore_then(var);
    let nvar = || {
        var.map(|v| NVar::new(v, false))
            .or(not_var().map(|v| NVar::new(v, true)))
    };
    let op_nexpr = |c, f| nvar().separated_by(op(c)).at_least(2).map(f);
    let op_expr = |c, f| var.separated_by(op(c)).at_least(2).map(f);
    let expr = not_var()
        .map(|v| Expr::Not(v))
        .or(ident
            .then(var.delimited_by(op('['), op(']')))
            .map(|(table, var)| Expr::Lookup { table, var }))
        .or(op_nexpr('^', Expr::Xor as fn(_) -> _))
        .or(op_nexpr('&', Expr::And as fn(_) -> _))
        .or(op_expr('+', Expr::Add as fn(_) -> _))
        .or(op_expr('*', Expr::Mul as fn(_) -> _));
    let prop = kw("PROPERTY")
        .ignore_then(var)
        .then_ignore(op('='))
        .then(expr)
        .map(|(dest, expr)| Statement::Property { dest, expr });
    let nc = kw("NC")
        .ignore_then(text::int(10))
        .map(|nc: String| Statement::NC(nc.parse().unwrap()));
    let comment = kw("#")
        .then_ignore(filter(|c| c != '\n' && c != '\r').repeated())
        .to(Statement::Comment);
    let var_decl = kw("VAR")
        .ignore_then(kw("SINGLE").to(false).or(kw("MULTI").to(true)))
        .then(var)
        .map(|(multi, name)| Statement::VarDecl(VarDecl { name, multi }));
    let pub_decl = kw("PUB")
        .ignore_then(kw("SINGLE").to(false).or(kw("MULTI").to(true)))
        .then(var)
        .map(|(multi, name)| Statement::PubDecl(VarDecl { name, multi }));
    let table_val = text::int(10)
        .map(|x: String| x.parse().unwrap())
        .separated_by(op(','))
        .allow_trailing()
        .delimited_by(op('['), op(']'));
    let table = kw("TABLE")
        .ignore_then(ident)
        .then(op('=').ignore_then(table_val).or_not())
        .map(|(name, val)| Statement::TableDecl { name, val });
    let graph = prop
        .or(nc)
        .or(var_decl)
        .or(table)
        .or(comment)
        .or(end().to(Statement::End))
        .recover_with(skip_until(['\n', '\r'], |_| Statement::Invalid))
        .separated_by(text::newline().then(text::whitespace().or_not()))
        .allow_leading()
        .allow_trailing();
    graph
}

fn parse_file() -> Result<Vec<Statement>, GraphError> {
    let src = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    println!("{:?}", src);
    let (graph, errs) = parser().parse_recovery_verbose(src.as_str());
    println!("{:#?}", graph);
    let err = !errs.is_empty();
    for e in errs {
        let msg = if let chumsky::error::SimpleReason::Custom(msg) = e.reason() {
            msg.clone()
        } else {
            format!(
                "Unexpected token, expected {}",
                if e.expected().len() == 0 {
                    "something else".to_string()
                } else {
                    e.expected()
                        .map(|expected| match expected {
                            Some(expected) => expected.to_string(),
                            None => "end of input".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                },
            )
        };
        let report = Report::build(ReportKind::Error, (), e.span().start)
            .with_code(3)
            .with_message(msg)
            .with_label(
                Label::new(e.span())
                    .with_message(match e.reason() {
                        chumsky::error::SimpleReason::Custom(msg) => msg.clone(),
                        _ => format!(
                            "Unexpected {}",
                            e.found()
                                .map(|c| format!("token {}", c.fg(Color::Red)))
                                .unwrap_or_else(|| "end of input".to_string())
                        ),
                    })
                    .with_color(Color::Red),
            );
        report.finish().print(Source::from(&src)).unwrap();
    }
    if err {
        return Err(GraphError::TODO);
    } else {
        return Ok(graph.unwrap());
    }
}

type NamedList<T> = IndexMap<String, T>;

fn build_graph(
    stmts: &[Statement],
    tables: NamedList<Vec<ClassVal>>,
) -> Result<fg::FactorGraph, GraphError> {
    let nc = get_nc(stmts)?;
    let mut graph = fg::FactorGraph::build(nc);
    for s in stmts {
        match s {
            Statement::VarDecl(vd) => {
                graph.add_var(vd.name.0.clone(), vd.multi)?;
            }
            Statement::PubDecl(vd) => {
                graph.add_pub(vd.name.0.clone(), vd.multi)?;
            }
            Statement::TableDecl { name, val } => {
                let val = if let Some(val) = val {
                    val.clone()
                } else if let Some(val) = tables.get_mut(name) {
                    std::mem::replace(val, Vec::new())
                } else {
                    return Err(GraphError::MissingTableDef(name.clone()));
                };
                graph.add_table(name.clone(), val)?;
            }
            Statement::End
            | Statement::Invalid
            | Statement::Comment
            | Statement::Property
            | Statement::NC => {}
        }
    }
    for s in stmts {
        if let Statement::Property { dest, expr } = s {
            graph.add_factor(
                expr.as_factor_kind(),
                std::iter::once(dest.0.as_str()).chain(expr.vars()),
            )?;
        }
    }
    Ok(graph)
}

fn get_nc(stmts: &[Statement]) -> Result<u64, GraphError> {
    let mut nc_decls = stmts.iter().filter_map(|s| {
        if let Statement::NC(nc) = s {
            Some(*nc)
        } else {
            None
        }
    });
    if let Some(nc) = nc_decls.next() {
        if nc_decls.next().is_some() {
            return Err(GraphError::MultipleNc);
        } else {
            Ok(nc)
        }
    } else {
        return Err(GraphError::NoNc);
    }
}

fn main() {
    if let Ok(graph) = parse_file() {}
}
