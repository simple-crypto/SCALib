pub mod belief_propagation;
pub mod lda;
pub(crate) mod matrixmul;
pub mod mttest;
pub mod sasca;
pub mod snr;
pub mod ttest;
pub(crate) mod utils;

use thiserror::Error;

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
}

#[derive(Clone)]
pub struct Config {
    /// Show progress bars
    pub show_progress: bool,
    /// Computation time after which a progress bar is displayed.
    /// This avoids showing progress bars for negligible amounts of time.
    pub progress_min_time: std::time::Duration,
}

impl std::default::Default for Config {
    fn default() -> Self {
        Self {
            show_progress: true,
            progress_min_time: std::time::Duration::from_millis(500),
        }
    }
}
