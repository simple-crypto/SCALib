
mod belief_propabation;
mod bp_compute;
mod factor_graph;
mod fg_parser;
mod fg_build;


pub use bp_compute::Distribution;

pub type ClassVal = u32;
type NamedList<T> = indexmap::IndexMap<String, T>;

pub use factor_graph::{FactorGraph, PublicValue, VarId, FactorId, EdgeId};
pub use fg_build::GraphBuildError;
pub use belief_propabation::{BPState, BPError};


pub fn build_graph(description: &str, tables: std::collections::HashMap<String, Vec<ClassVal>>) -> Result<FactorGraph, GraphBuildError> {
    let stmts = fg_parser::parse(description).map_err(|e| GraphBuildError::Parse(String::from_utf8_lossy(e.as_slice()).into_owned()))?;
    fg_build::build_graph(stmts.as_slice(), tables)
}
