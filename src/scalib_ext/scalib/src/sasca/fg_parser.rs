use super::ClassVal;
use ariadne::{Color, Fmt, Label, Report, ReportKind, Source};
use chumsky::prelude::*;

#[derive(Debug, Clone)]
pub(super) struct Var(pub(super) String);

#[derive(Debug, Clone)]
pub(super) struct NVar {
    pub(super) var: Var,
    pub(super) neg: bool,
}

impl NVar {
    fn new(var: Var, neg: bool) -> Self {
        Self { var, neg }
    }
}

#[derive(Debug, Clone)]
pub(super) enum Expr {
    Not(Var),
    Lookup { var: Var, table: String },
    Add(Vec<Var>),
    Mul(Vec<Var>),
    Xor(Vec<Var>),
    And(Vec<NVar>),
    Or(Vec<NVar>),
}

#[derive(Debug, Clone)]
pub(super) struct VarDecl {
    pub(super) name: Var,
    pub(super) multi: bool,
}

#[derive(Debug, Clone)]
pub(super) enum Statement {
    Invalid,
    Empty,
    Property {
        name: Option<String>,
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
    let expr = ident
        .then(var.delimited_by(op('['), op(']')))
        .map(|(table, var)| Expr::Lookup { table, var })
        .or(op_expr('^', Expr::Xor as fn(_) -> _))
        .or(op_nexpr('&', Expr::And as fn(_) -> _))
        .or(op_nexpr('|', Expr::Or as fn(_) -> _))
        .or(op_expr('+', Expr::Add as fn(_) -> _))
        .or(op_expr('*', Expr::Mul as fn(_) -> _))
        .or(not_var().map(|v| Expr::Not(v)));
    let prop = kw("PROPERTY")
        .ignore_then(ident.then_ignore(op(':')).or_not())
        .then(var)
        .then_ignore(op('='))
        .then(expr)
        .map(|((name, dest), expr)| Statement::Property { name, dest, expr });
    let nc = kw("NC")
        .ignore_then(text::int(10))
        .map(|nc: String| Statement::NC(nc.parse().unwrap()));
    let comment = op('#').then_ignore(filter(|c| *c != '\n' && *c != '\r').repeated());
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
        .or(pub_decl)
        .or(table)
        .or(pad.at_least(0).to(Statement::Empty))
        .then_ignore(comment.or_not())
        .recover_with(skip_until(['\n', '\r'], |_| Statement::Invalid))
        .separated_by(text::newline())
        .allow_leading()
        .allow_trailing()
        .then_ignore(end());
    graph
}

/// Parse the factor graph description in src, and returns the statements if no
/// error, otherwise the error is a locale-encoded error message.
pub(super) fn parse(src: &str) -> Result<Vec<Statement>, Vec<u8>> {
    let (graph, errs) = parser().parse_recovery_verbose(src);
    let err = !errs.is_empty();
    let mut err_str = Vec::new();
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
        report
            .finish()
            .write(Source::from(&src), &mut err_str)
            .unwrap();
    }
    if err {
        return Err(err_str);
    } else {
        return Ok(graph.unwrap());
    }
}
