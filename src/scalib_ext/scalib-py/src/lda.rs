//! Python binding of SCALib's LDA implementation.

use crate::ScalibError;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct LdaAcc {
    inner: scalib::lda::LdaAcc,
}
#[pymethods]
impl LdaAcc {
    #[new]
    /// Init an LDA empty LDA accumulator
    fn new(nc: usize, ns: usize) -> Self {
        Self {
            inner: scalib::lda::LdaAcc::from_dim(nc, ns),
        }
    }
    /// Add measurements to the accumulator
    /// x: traces with shape (n,ns)
    /// y: random value realization (n,)
    /// gemm_algo is 0 for ndarray gemm, x>0 for BLIS gemm with x threads.
    fn fit(
        &mut self,
        py: Python,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray1<u16>,
        gemm_algo: u32,
        config: crate::ConfigWrapper,
    ) {
        let x = x.as_array();
        let y = y.as_array();
        config.on_worker(py, |_| self.inner.update(x, y, gemm_algo));
    }

    /// Compute the LDA with p dimensions in the projected space
    fn lda(&self, py: Python, p: usize, config: crate::ConfigWrapper) -> PyResult<LDA> {
        match config.on_worker(py, |_| self.inner.lda(p)) {
            Ok(inner) => Ok(LDA { inner }),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Get the state for serialization
    fn get_state<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        usize,
        usize,
        usize,
        &'py PyArray2<f64>,
        &'py PyArray2<f64>,
        &'py PyArray1<f64>,
        &'py PyArray1<usize>,
    ) {
        (
            self.inner.ns,
            self.inner.nc,
            self.inner.n,
            self.inner.scatter.to_pyarray(py),
            self.inner.traces_sum.to_pyarray(py),
            self.inner.mu.to_pyarray(py),
            self.inner.n_traces.to_pyarray(py),
        )
    }

    /// Get the matrix sw (debug purpose)
    fn get_sw<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        match self.inner.get_matrices() {
            Ok((sw, _, _)) => Ok(sw.into_pyarray(py)),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Get the matrix sb (debug purpose)
    fn get_sb<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        match self.inner.get_matrices() {
            Ok((_, sb, _)) => Ok(sb.into_pyarray(py)),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Get the matrix mus (debug purpose)
    fn get_mus<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        match self.inner.get_matrices() {
            Ok((_, _, mus)) => Ok(mus.into_pyarray(py)),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    /// Set the accumulator state
    #[staticmethod]
    fn from_state(
        ns: usize,
        nc: usize,
        n: usize,
        scatter: PyReadonlyArray2<f64>,
        traces_sum: PyReadonlyArray2<f64>,
        mu: PyReadonlyArray1<f64>,
        n_traces: PyReadonlyArray1<usize>,
    ) -> Self {
        let mut inner = scalib::lda::LdaAcc::from_dim(nc, ns);
        inner.n = n;
        inner.scatter.assign(&scatter.as_array());
        inner.traces_sum.assign(&traces_sum.as_array());
        inner.mu.assign(&mu.as_array());
        inner.n_traces.assign(&n_traces.as_array());
        Self { inner }
    }
}

#[pyclass]
pub(crate) struct LDA {
    inner: scalib::lda::LDA,
}
#[pymethods]
impl LDA {
    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (n,nc). Every row corresponds to one probability distribution
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        config: crate::ConfigWrapper,
    ) -> PyResult<&'py PyArray2<f64>> {
        let x = x.as_array();
        let prs = config.on_worker(py, |_| self.inner.predict_proba(x));
        Ok(prs.to_pyarray(py))
    }

    /// Get the lda state for serialization
    fn get_state<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        &'py PyArray2<f64>,
        usize,
        usize,
        usize,
        &'py PyArray2<f64>,
        &'py PyArray1<f64>,
    ) {
        (
            self.inner.projection.to_pyarray(py),
            self.inner.ns,
            self.inner.p,
            self.inner.nc,
            self.inner.omega.to_pyarray(py),
            self.inner.pk.to_pyarray(py),
        )
    }

    /// Get the projection matrix
    fn get_projection<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.inner.projection.to_pyarray(py)
    }

    /// Set the LDA state based on all its parameters
    #[staticmethod]
    fn from_state(
        projection: PyReadonlyArray2<f64>,
        ns: usize,
        p: usize,
        nc: usize,
        omega: PyReadonlyArray2<f64>,
        pk: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            inner: scalib::lda::LDA {
                projection: projection.to_owned_array(),
                ns,
                p,
                nc,
                omega: omega.to_owned_array(),
                pk: pk.to_owned_array(),
            },
        }
    }
}
