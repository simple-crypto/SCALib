pub mod belief_propagation;
pub mod information;
pub mod lda;
pub(crate) mod matrixmul;
pub mod mttest;
pub mod multi_lda;
pub mod rlda;
pub mod sasca;
pub mod snr;
pub mod ttest;
pub(crate) mod utils;

use thiserror::Error;

type Result<T> = std::result::Result<T, ScalibError>;

#[derive(Error, Debug)]
pub enum ScalibError {
    #[error("A class is missing data.")]
    EmptyClass,
    #[error("An error occured in the linear algebra solver.")]
    GeigenError(#[from] geigen::GeigenError),
    #[error("Too many traces for SNR (>= 2**32).")]
    SnrTooManyTraces,
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
    #[error("Empty KdTree, cannot get nearest centroid")]
    EmptyKdTree,
    #[error("No associated classes stored")]
    NoAssociatedClassesStored,
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
