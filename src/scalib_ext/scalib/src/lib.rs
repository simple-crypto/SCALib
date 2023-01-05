pub mod belief_propagation;
pub mod lda;
pub(crate) mod matrixmul;
pub mod mttest;
pub mod snr;
pub mod ttest;

use thiserror::Error;
#[derive(Error, Debug)]
pub enum ScalibError {
    #[error("A class is missing data.")]
    EmptyClass,
    #[error("An error occured in the linear algebra solver.")]
    GeigenError(#[from] geigen::GeigenError),
    #[error("Too many traces for SNR (>= 2**32).")]
    SnrTooManyTraces,
    #[error("Too many traces for SNR, try setting use_64bit=True.")]
    SnrClassOverflow,
    #[error("A SNR class value of a variable is larger than the given number of classes.")]
    SnrClassOutOfBound,
}
