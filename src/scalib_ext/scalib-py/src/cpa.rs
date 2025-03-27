//! Python binding of SCALib's CPA implementation.

use crate::ScalibError;
use numpy::{PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use scalib::cpa;

enum InnerCpa {
    Cpa32bit(cpa::CPA<scalib::AccType32bit>),
    Cpa64bit(cpa::CPA<scalib::AccType64bit>),
}

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct CPA {
    inner: InnerCpa,
}
#[pymethods]
impl CPA {
    #[new]
    /// Create a new CPA state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which CPA must be estimated
    fn new(nc: usize, ns: usize, np: usize, use_64bit: bool) -> Self {
        Self {
            inner: if use_64bit {
                InnerCpa::Cpa64bit(cpa::CPA::new(nc, ns, np))
            } else {
                InnerCpa::Cpa32bit(cpa::CPA::new(nc, ns, np))
            },
        }
    }
    /// Update the CPA state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    fn update(
        &mut self,
        py: Python,
        traces: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u16>,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        let inner = &mut self.inner;
        let x = traces.as_array();
        let y = y.as_array();
        config
            .on_worker(py, |cfg| match inner {
                InnerCpa::Cpa32bit(inner) => inner.update(x, y, cfg),
                InnerCpa::Cpa64bit(inner) => inner.update(x, y, cfg),
            })
            .map_err(|e| ScalibError::from_scalib(e, py))?;
        Ok(())
    }

    /// The actual CPA based on the current state.
    /// return array axes (variable, class, samples in trace)
    fn compute_cpa<'py>(
        &mut self,
        py: Python<'py>,
        models: PyReadonlyArray3<f64>,
        config: crate::ConfigWrapper,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let models = models.as_array();
        let cpa = config
            .on_worker(py, |_| match &self.inner {
                InnerCpa::Cpa32bit(inner) => inner.compute_cpa(models),
                InnerCpa::Cpa64bit(inner) => inner.compute_cpa(models),
            })
            .map_err(|e| ScalibError::from_scalib(e, py))?;
        Ok(cpa.to_pyarray(py))
    }
}
