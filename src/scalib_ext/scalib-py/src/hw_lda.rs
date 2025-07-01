//! Python binding of SCALib's MultiLda implementation.

use bincode::{deserialize, serialize};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyBytes, PyTuple};

use crate::ScalibError;

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct HwLdaAcc {
    inner: scalib::hw_lda::HwLdaAcc,
}
#[pymethods]
impl HwLdaAcc {
    #[new]
    /// Init an LDA empty LDA accumulator
    fn new(py: Python, nb: u32, ns: u32, nv: u32) -> PyResult<Self> {
        Ok(Self {
            inner: scalib::hw_lda::HwLdaAcc::new(nb, ns, nv),
        })
    }
    /// Add measurements to the accumulator
    /// x: traces with shape (n,ns)
    /// y: random value realization (n,nv)
    fn fit(
        &mut self,
        py: Python,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u64>,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        let x = x.as_array();
        let y = y.as_array();
        config.on_worker(py, |_| self.inner.update(x, y, 0));
        Ok(())
    }

    fn lda(&self, py: Python, config: crate::ConfigWrapper) -> PyResult<HwLda> {
        match config.on_worker(py, |cfg| self.inner.solve()) {
            Ok(inner) => Ok(HwLda { inner }),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                Self::type_object(py).getattr(intern!(py, "_from_bytes"))?,
                PyTuple::new(
                    py,
                    [PyBytes::new(py, &serialize(&self.inner).unwrap()).into_any()],
                )?
                .into_any(),
            ],
        )
    }

    #[staticmethod]
    fn _from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: deserialize(bytes).map_err(|_| PyValueError::new_err("Invalid state bytes."))?,
        })
    }
}

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct HwLda {
    inner: scalib::hw_lda::HwLda,
}
#[pymethods]
impl HwLda {
    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// v : id of variable
    /// return prs with shape (n,nc). Each last-axis view corresponds to one probability distribution.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        traces: PyReadonlyArray2<i16>,
        v: u32,
        config: crate::ConfigWrapper,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = traces.as_array();
        let prs = config.on_worker(py, |_| self.inner.predict_proba(x, v));
        Ok(prs.into_pyarray(py))
    }
    fn predict_log2_proba_class<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u64>,
        config: crate::ConfigWrapper,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = x.as_array();
        let y = y.as_array();
        let prs = config.on_worker(py, |_| self.inner.predict_log2p1(x, y));
        Ok(prs.into_pyarray(py))
    }
    fn project<'py>(
        &self,
        py: Python<'py>,
        traces: PyReadonlyArray2<i16>,
        v: u32,
        config: crate::ConfigWrapper,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let traces = traces.as_array();
        let p_traces = config.on_worker(py, |_| self.inner.project(traces.view(), v));
        Ok(p_traces.into_pyarray(py))
    }
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                Self::type_object(py).getattr(intern!(py, "_from_bytes"))?,
                PyTuple::new(
                    py,
                    [PyBytes::new(py, &serialize(&self.inner).unwrap()).into_any()],
                )?
                .into_any(),
            ],
        )
    }

    #[staticmethod]
    fn _from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: deserialize(bytes).map_err(|_| PyValueError::new_err("Invalid state bytes."))?,
        })
    }
}
