mod belief_propagation;
mod bp_compute;
mod factor_graph;
mod fg_build;
mod fg_parser;

pub use bp_compute::Distribution;

pub type ClassVal = u32;
type NamedList<T> = indexmap::IndexMap<String, T>;

pub use belief_propagation::{BPError, BPState};
pub use factor_graph::{EdgeId, FactorGraph, FactorId, PublicValue, VarId};
pub use fg_build::GraphBuildError;

pub fn build_graph(
    description: &str,
    tables: std::collections::HashMap<String, Vec<ClassVal>>,
) -> Result<FactorGraph, GraphBuildError> {
    let stmts = fg_parser::parse(description)
        .map_err(|e| GraphBuildError::Parse(String::from_utf8_lossy(e.as_slice()).into_owned()))?;
    fg_build::build_graph(stmts.as_slice(), tables)
}
