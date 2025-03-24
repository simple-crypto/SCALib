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
pub(crate) struct MultiLdaAcc {
    inner: scalib::multi_lda::MultiLdaAcc,
}
#[pymethods]
impl MultiLdaAcc {
    #[new]
    /// Init an LDA empty LDA accumulator
    fn new(py: Python, ns: u32, nc: u16, pois: Vec<Vec<u32>>) -> PyResult<Self> {
        Ok(Self {
            inner: scalib::multi_lda::MultiLdaAcc::new(ns, nc, pois)
                .map_err(|e| ScalibError::from_scalib(e, py))?,
        })
    }
    /// Add measurements to the accumulator
    /// x: traces with shape (n,ns)
    /// y: random value realization (n,)
    fn fit(
        &mut self,
        py: Python,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u16>,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        let x = x.as_array();
        let y = y.as_array();
        config
            .on_worker(py, |_| self.inner.update(x, y))
            .map_err(|e| ScalibError::from_scalib(e, py))
    }

    fn multi_lda(&self, py: Python, p: u32, config: crate::ConfigWrapper) -> PyResult<MultiLda> {
        match config.on_worker(py, |cfg| self.inner.lda(p, cfg)) {
            Ok(inner) => Ok(MultiLda { inner }),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Get the matrix sw
    fn get_sw<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        match self.inner.get_matrices() {
            Ok(m) => Ok(m
                .into_iter()
                .map(|(sw, _, _)| sw.into_pyarray(py))
                .collect()),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }
    /// Get the matrix sb

    fn get_sb<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        match self.inner.get_matrices() {
            Ok(m) => Ok(m
                .into_iter()
                .map(|(_, sb, _)| sb.into_pyarray(py))
                .collect()),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Get the matrix mus
    fn get_mus<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        match self.inner.get_matrices() {
            Ok(m) => Ok(m
                .into_iter()
                .map(|(_, _, mus)| mus.into_pyarray(py))
                .collect()),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    fn get_matrices<'py>(
        &self,
        py: Python<'py>,
        config: crate::ConfigWrapper,
    ) -> PyResult<
        Vec<(
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray2<f64>>,
        )>,
    > {
        Ok(config
            .on_worker(py, |_| self.inner.get_matrices())
            .map_err(|e| ScalibError::from_scalib(e, py))?
            .into_iter()
            .map(|(sw, sb, mus)| {
                (
                    sw.into_pyarray(py),
                    sb.into_pyarray(py),
                    mus.into_pyarray(py),
                )
            })
            .collect())
    }

    fn n_traces(&self) -> u32 {
        self.inner.ntraces()
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
pub(crate) struct MultiLda {
    inner: scalib::multi_lda::MultiLda,
}
#[pymethods]
impl MultiLda {
    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (nv,n,nc). Each last-axis view corresponds to one probability distribution.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        config: crate::ConfigWrapper,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let x = x.as_array();
        let prs = config.on_worker(py, |_| self.inner.predict_proba(x));
        Ok(prs.into_pyarray(py))
    }
    fn predict_log2_proba_class<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u16>,
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
        config: crate::ConfigWrapper,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let traces = traces.as_array();
        let p_traces = config.on_worker(py, |_| self.inner.project(traces.view()));
        Ok(p_traces.into_iter().map(|a| a.into_pyarray(py)).collect())
    }
    fn select_vars(&self, py: Python, vars: Vec<u16>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .select_vars(&vars)
                .map_err(|e| ScalibError::from_scalib(e, py))?,
        })
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
