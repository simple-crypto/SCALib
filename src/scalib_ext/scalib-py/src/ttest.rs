//! Python wrapper for SCALib's Ttest

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
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
    fn update(
        &mut self,
        py: Python,
        traces: PyReadonlyArray2<i16>,
        y: PyReadonlyArray1<u16>,
        config: crate::ConfigWrapper,
    ) {
        let traces = traces.as_array();
        let y = y.as_array();
        config.on_worker(py, |_| self.inner.update(traces, y));
    }

    /// Generate the actual Ttest metric based on the current state.
    /// return array axes (d,ns)
    fn get_ttest<'py>(
        &mut self,
        py: Python<'py>,
        config: crate::ConfigWrapper,
    ) -> PyResult<&'py PyArray2<f64>> {
        let ttest = config.on_worker(py, |_| self.inner.get_ttest());
        Ok(ttest.to_pyarray(py))
    }
}

#[pyclass]
pub(crate) struct MTtest {
    inner: scalib::mttest::MTtest,
}

#[pymethods]
impl MTtest {
    #[new]
    /// Create a new Ttest state.
    /// d: order of the Ttest
    /// pois: points of interest
    fn new(d: usize, pois: PyReadonlyArray2<u32>) -> Self {
        let pois = pois.as_array();
        Self {
            inner: scalib::mttest::MTtest::new(d, pois.view()),
        }
    }
    /// Update the Ttest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    fn update(
        &mut self,
        py: Python,
        traces: PyReadonlyArray2<i16>,
        y: PyReadonlyArray1<u16>,
        config: crate::ConfigWrapper,
    ) {
        let traces = traces.as_array();
        let y = y.as_array();
        config.on_worker(py, |_| self.inner.update(traces, y));
    }

    /// Generate the actual Ttest metric based on the current state.
    /// return array axes (d,ns)
    fn get_ttest<'py>(
        &mut self,
        py: Python<'py>,
        config: crate::ConfigWrapper,
    ) -> PyResult<&'py PyArray1<f64>> {
        let ttest = config.on_worker(py, |_| self.inner.get_ttest());
        Ok(ttest.to_pyarray(py))
    }
}
