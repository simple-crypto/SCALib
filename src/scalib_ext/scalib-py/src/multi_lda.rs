//! Python binding of SCALib's MultiLda implementation.

use crate::ScalibError;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
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
    /// gemm_algo is 0 for ndarray gemm, x>0 for BLIS gemm with x threads.
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

    /// Compute the LDA with p dimensions in the projected space
    fn lda(
        &self,
        py: Python,
        p: usize,
        config: crate::ConfigWrapper,
    ) -> PyResult<Vec<crate::lda::LDA>> {
        config
            .on_worker(py, |_| {
                let n = self.inner.ntraces() as usize;
                let res = self
                    .inner
                    .get_matrices()?
                    .into_iter()
                    .map(|(sw, sb, mus)| {
                        Ok(crate::lda::LDA {
                            inner: scalib::lda::LDA::from_matrices(
                                n,
                                p,
                                sw.view(),
                                sb.view(),
                                mus.view(),
                            )?,
                        })
                    })
                    .collect::<Result<Vec<crate::lda::LDA>, _>>();
                res
            })
            .map_err(|e| ScalibError::from_scalib(e, py))
    }

    /// Get the matrix sw (debug purpose)
    fn get_sw<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        match self.inner.get_matrices() {
            Ok(m) => Ok(m
                .into_iter()
                .map(|(sw, _, _)| sw.into_pyarray(py))
                .collect()),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }
    /// Get the matrix sb (debug purpose)

    fn get_sb<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        match self.inner.get_matrices() {
            Ok(m) => Ok(m
                .into_iter()
                .map(|(_, sb, _)| sb.into_pyarray(py))
                .collect()),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Get the matrix mus (debug purpose)
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
}
