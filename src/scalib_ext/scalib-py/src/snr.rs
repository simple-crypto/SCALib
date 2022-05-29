//! Python binding of SCALib's SNR implementation.

use crate::thread_pool::ThreadPool;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use scalib::snr;

create_exception!(_scalib_ext, SnrError, PyException);

enum InnerSnr {
    Snr32bit(snr::SNR<snr::SnrType32bit>),
    Snr64bit(snr::SNR<snr::SnrType64bit>),
}

#[pyclass]
pub(crate) struct SNR {
    inner: InnerSnr,
}
#[pymethods]
impl SNR {
    #[new]
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which SNR must be estimated
    fn new(nc: usize, ns: usize, np: usize, use_64bit: bool) -> Self {
        Self {
            inner: if use_64bit {
                InnerSnr::Snr64bit(snr::SNR::new(nc, ns, np))
            } else {
                InnerSnr::Snr32bit(snr::SNR::new(nc, ns, np))
            },
        }
    }
    /// Update the SNR state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    fn update(
        &mut self,
        py: Python,
        traces: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u16>,
        thread_pool: &ThreadPool,
    ) -> PyResult<()> {
        let inner = &mut self.inner;
        let x = traces.as_array();
        let y = y.as_array();
        crate::on_worker(py, thread_pool, || match inner {
            InnerSnr::Snr32bit(inner) => inner.update(x, y),
            InnerSnr::Snr64bit(inner) => inner.update(x, y),
        })
        .map_err(|e| SnrError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Generate the actual SNR metric based on the current state.
    /// return array axes (variable, samples in trace)
    fn get_snr<'py>(
        &mut self,
        py: Python<'py>,
        thread_pool: &ThreadPool,
    ) -> PyResult<&'py PyArray2<f64>> {
        let snr = crate::on_worker(py, thread_pool, || match &self.inner {
            InnerSnr::Snr32bit(inner) => inner.get_snr(),
            InnerSnr::Snr64bit(inner) => inner.get_snr(),
        });
        Ok(snr.to_pyarray(py))
    }
}
