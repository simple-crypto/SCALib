//! Python binding of SCALib's SNR implementation.

use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct SNR {
    inner: scalib::snr::SNR,
}
#[pymethods]
impl SNR {
    #[new]
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which SNR must be estimated
    fn new(nc: usize, ns: usize, np: usize) -> Self {
        Self {
            inner: scalib::snr::SNR::new(nc, ns, np),
        }
    }
    /// Update the SNR state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    fn update(&mut self, py: Python, traces: PyReadonlyArray2<i16>, y: PyReadonlyArray2<u16>) {
        // FIXME I believe we are missing a lock here... is this a bug of PyO3
        let inner = &mut self.inner;
        let x = traces.as_array();
        let y = y.as_array();
        py.allow_threads(|| inner.update(x, y));
    }

    /// Generate the actual SNR metric based on the current state.
    /// return array axes (variable, samples in trace)
    fn get_snr<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let snr = py.allow_threads(|| self.inner.get_snr());
        Ok(&(snr.to_pyarray(py)))
    }
}
