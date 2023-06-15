//! Python binding of SCALib's RLDA implementation.

use crate::ScalibError;
use bincode::{deserialize, serialize};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use std::sync::Arc;

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct RLDA {
    inner: Option<scalib::rlda::RLDA>,
}
#[pymethods]
impl RLDA {
    /// Init an empty RLDA model
    #[new]
    #[pyo3(signature = (*args))]
    fn new(_py: Python, args: &PyTuple) -> PyResult<Self> {
        if args.len() == 0 {
            Ok(Self { inner: None })
        } else {
            let (nb, ns, nv, p): (usize, usize, usize, usize) = args.extract()?;
            let inner = Some(scalib::rlda::RLDA::new(nb, ns, nv, p));
            Ok(Self { inner })
        }
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    /// Add n measurements to the model
    /// x: traces with shape (n,ns)
    /// y: random value realization (nv,n)
    /// gemm_algo is 0 for ndarray gemm, x>0 for BLIS gemm with x threads.
    fn update(
        &mut self,
        py: Python,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u64>,
        gemm_algo: u32,
        config: crate::ConfigWrapper,
    ) {
        let x = x.as_array();
        let y = y.as_array();
        config.on_worker(py, |_| self.inner.as_mut().unwrap().update(x, y, gemm_algo));
    }

    fn solve<'py>(&mut self, py: Python<'py>, config: crate::ConfigWrapper) -> PyResult<()> {
        config
            .on_worker(py, |_| self.inner.as_mut().unwrap().solve())
            .map_err(|e| ScalibError::from_scalib(e, py))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        v: usize,
        config: crate::ConfigWrapper,
    ) -> PyResult<&'py PyArray2<f64>> {
        let x = x.as_array();
        let prs = config.on_worker(py, |_| self.inner.as_ref().unwrap().predict_proba(x, v));
        Ok(prs.into_pyarray(py))
    }

    fn get_proj_coefs<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<f64>> {
        Ok(self.inner.as_ref().unwrap().proj_coefs.to_pyarray(py))
    }

    fn get_norm_proj<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<f64>> {
        Ok(self.inner.as_ref().unwrap().norm_proj.to_pyarray(py))
    }

    fn get_clustered_model<'py>(
        &self,
        py: Python<'py>,
        var_id: usize,
        store_associated_classes: bool,
        max_distance: f64,
        max_cluster_number: u32,
    ) -> PyResult<RLDAClusteredModel> {
        match self.inner.as_ref().unwrap().get_clustered_model(
            var_id,
            store_associated_classes,
            max_distance,
            max_cluster_number,
        ) {
            Ok(inner) => Ok(RLDAClusteredModel {
                inner: Some(Arc::new(inner)),
            }),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }
}

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct RLDAClusteredModel {
    pub inner: Option<Arc<scalib::rlda::RLDAClusteredModel>>,
}

#[pymethods]
impl RLDAClusteredModel {
    #[new]
    fn new(_py: Python) -> PyResult<Self> {
        Ok(Self { inner: None })
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    // Not used by python code, debug purpose only.
    fn nearest<'_py>(
        &mut self,
        py: Python<'_py>,
        point: PyReadonlyArray1<f64>,
    ) -> PyResult<(usize, f64)> {
        self.inner
            .as_ref()
            .unwrap()
            .nearest(point.as_slice().unwrap())
            .map_err(|e| ScalibError::from_scalib(e, py))
    }

    // Not used by python code, debug purpose only.
    fn get_max_sqdist<'_py>(&mut self, _py: Python<'_py>) -> f64 {
        self.inner.as_ref().unwrap().max_squared_distance
    }

    // Not used by python code, debug purpose only.
    fn get_size<'_py>(&mut self, _py: Python<'_py>) -> u32 {
        self.inner.as_ref().unwrap().get_size()
    }

    // Not used by python code, debug purpose only.
    fn get_close_cluster_centers<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        max_n_points: usize,
    ) -> PyResult<&'py PyArray1<usize>> {
        match self
            .inner
            .as_ref()
            .unwrap()
            .get_close_cluster_centers(point.as_slice().unwrap(), max_n_points)
        {
            Ok(iterator) => Ok(iterator
                .map(|(c_id, _n_associated)| c_id)
                .collect::<Vec<usize>>()
                .to_pyarray(py)),
            Err(e) => Err(ScalibError::from_scalib(e, py)),
        }
    }

    fn get_bounded_prs<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        label: PyReadonlyArray1<u64>,
        max_popped_classes: usize,
        config: crate::ConfigWrapper,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>)> {
        let x = x.as_array();
        let label = label.as_array();
        let prs = config.on_worker(py, |_| {
            self.inner
                .as_ref()
                .unwrap()
                .bounded_prs(x, label, max_popped_classes)
        });
        Ok((prs.0.to_pyarray(py), prs.1.to_pyarray(py)))
    }
}
