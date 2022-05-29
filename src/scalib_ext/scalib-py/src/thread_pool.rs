use pyo3::create_exception;
use pyo3::exceptions;
use pyo3::prelude::*;
use rayon;

create_exception!(_scalib_ext, ThreadPoolError, exceptions::PyOSError);

#[pyclass]
pub struct ThreadPool {
    pub pool: rayon::ThreadPool,
}
#[pymethods]
impl ThreadPool {
    #[new]
    /// Create a new ThreadPool, with a given number of threads
    fn new(num_threads: usize) -> PyResult<Self> {
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| ThreadPoolError::new_err(e.to_string()))?,
        })
    }
}
