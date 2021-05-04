//! Python binding of SCALib's LDA implementation.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct LDA {
    inner: scalib::lda::LDA,
}
#[pymethods]
impl LDA {
    #[new]
    /// Init an LDA with empty arrays
    fn new(nc: usize, p: usize, ns: usize) -> Self {
        Self {
            inner: scalib::lda::LDA::new(nc, p, ns),
        }
    }
    /// Fit the LDA with measurements to derive projection,means,covariance and psd.
    /// x: traces with shape (n,ns)
    /// y: random value realization (n,)
    fn fit(
        &mut self,
        py: Python,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray1<u16>,
        eigen_mode: u8,
        cov_mode: u8,
        test_cov: bool,
    ) {
        let x = x.as_array();
        let y = y.as_array();
        py.allow_threads(|| self.inner.fit(x, y, eigen_mode, cov_mode, test_cov));
    }

    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (n,nc). Every row corresponds to one probability distribution
    fn predict_proba<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let prs = self.inner.predict_proba(x.as_array());
        Ok(&(prs.to_pyarray(py)))
    }

    /// Set the LDA state based on all its parameters
    fn set_state<'py>(
        &mut self,
        _py: Python<'py>,
        cov: PyReadonlyArray2<f64>,
        psd: PyReadonlyArray2<f64>,
        means: PyReadonlyArray2<f64>,
        projection: PyReadonlyArray2<f64>,
        nc: usize,
        p: usize,
        ns: usize,
    ) {
        self.inner.cov.assign(&cov.as_array());
        self.inner.psd.assign(&psd.as_array());
        self.inner.means.assign(&means.as_array());
        self.inner.projection.assign(&projection.as_array());
        self.inner.nc = nc;
        self.inner.p = p;
        self.inner.ns = ns;
    }

    /// Get LDA internal data
    fn get_cov<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.inner.cov.to_pyarray(py))
    }
    fn get_projection<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.inner.projection.to_pyarray(py))
    }
    fn get_means<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.inner.means.to_pyarray(py))
    }
    fn get_psd<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.inner.psd.to_pyarray(py))
    }
}
