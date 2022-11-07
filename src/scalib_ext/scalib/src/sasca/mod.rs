
mod bp_compute;
mod factor_graph;
mod fg_parser;
mod fg_build;

pub use bp_compute::Distribution;

type ClassVal = u32;
type NamedList<T> = indexmap::IndexMap<String, T>;
