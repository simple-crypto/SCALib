pub mod belief_propagation;
pub mod cpa;
pub mod information;
pub mod lda;
pub mod lvar; // pub required for benches
pub(crate) mod matrixmul;
pub mod mttest;
pub mod rlda;
pub mod sasca;
pub mod snr;
pub mod ttest;
pub(crate) mod utils;

// Let us make some conservative assuptions regarding cache sizes
// - L2 = 512 kB
// - L3/ncores = 2 MB
const L2_SIZE: usize = 512 * 1024;
const L3_CORE: usize = 2 * 1024 * 1024;

pub use lvar::{AccType32bit, AccType64bit};

use thiserror::Error;

type Result<T> = std::result::Result<T, ScalibError>;

#[derive(Error, Debug)]
pub enum ScalibError {
    #[error("A class is missing data.")]
    EmptyClass,
    #[error("An error occured in the linear algebra solver.")]
    GeigenError(#[from] geigen::GeigenError),
    #[error("Too many traces (>= 2**32).")]
    TooManyTraces,
    #[error(
        "Too many traces for SNR, a sum might overflow. Set use_64bit=True to solve this. \
         (There is a class with {max_n_traces} traces, upper bound for abs. leakage value: {leak_upper_bound}."
    )]
    SnrClassOverflow {
        leak_upper_bound: i64,
        max_n_traces: i64,
    },
    #[error("A SNR class value of a variable is larger than the given number of classes.")]
    SnrClassOutOfBound,
    #[error("Clustering failed due to maximum number of centroids reached.")]
    MaxCentroidNumber,
    #[error("Empty KdTree, cannot get nearest centroid.")]
    EmptyKdTree,
    #[error("No associated classes stored.")]
    NoAssociatedClassesStored,
    #[error("Too many POIs.")]
    TooManyPois,
    #[error("Too many variables.")]
    TooManyVars,
    #[error("POI out of bounds.")]
    PoiOutOfBound,
    #[error("Variable out of bounds.")]
    VarOutOfBound,
    #[error("Incorrect shape for provided models: expected {expected:?}, got {dim:?}.")]
    CpaMShape {
        dim: (usize, usize, usize),
        expected: (usize, usize, usize),
    },
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Computation time after which a progress bar is displayed.
    /// This avoids showing progress bars for negligible amounts of time.
    /// If None, never display the progress bar
    progress_min_time: Option<std::time::Duration>,
}

impl Config {
    pub fn with_default_timing() -> Self {
        Self {
            progress_min_time: Some(std::time::Duration::from_millis(500)),
        }
    }
    pub fn no_progress() -> Self {
        Self {
            progress_min_time: None,
        }
    }
}
