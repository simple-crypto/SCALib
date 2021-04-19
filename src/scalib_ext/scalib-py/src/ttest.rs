//! Python wrapper for SCALib's Ttest

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct Ttest {
    inner: scalib::ttest::Ttest,
}

#[pymethods]
impl Ttest {
    #[new]
    /// Create a new Ttest state.
    /// ns: traces length
    /// d: order of the Ttest
    fn new(ns: usize, d: usize) -> Self {
        Self {
            inner: scalib::ttest::Ttest::new(ns, d),
        }
    }
    /// Update the Ttest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    fn update(&mut self, py: Python, traces: PyReadonlyArray2<i16>, y: PyReadonlyArray1<u16>) {
        let traces = traces.as_array();
        let y = y.as_array();
        py.allow_threads(|| self.inner.update(traces, y));
    }

    /// Generate the actual Ttest metric based on the current state.
    /// return array axes (d,ns)
    fn get_ttest<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let ttest = py.allow_threads(|| self.inner.get_ttest());
        Ok(&(ttest.to_pyarray(py)))
    }
}
