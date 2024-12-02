use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use scalib::{information::LDAModel, lda::LDA};
use std::sync::Arc;
use crate::rlda::RLDA;

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct ItEstimator {
    inner: scalib::information::ItEstimator,
}

#[pymethods]
impl ItEstimator {
    #[new]
    fn new(
        py: Python,
        model: &mut crate::rlda::RLDAClusteredModel,
        max_popped_classes: usize,
    ) -> PyResult<Self> {
        let inner = py.allow_threads(|| {
            scalib::information::ItEstimator::new(
                model.inner.as_ref().unwrap().clone(),
                max_popped_classes,
            )
        });
        Ok(Self { inner })
    }

    fn fit_u<'py>(
        &mut self,
        py: Python<'py>,
        traces: PyReadonlyArray2<i16>,
        label: PyReadonlyArray1<u64>,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let label = label.as_array();
        config.on_worker(py, |_| self.inner.fit_u(traces, label));
        Ok(())
    }

    fn get_information(&self) -> (f64, f64) {
        self.inner.get_information()
    }

    fn get_deviation(&self) -> (f64, f64, usize) {
        self.inner.get_deviation()
    }
}

pub(crate) struct LDAModel {
    ldamodel: scalib::information::LDAModel
}
impl LDAModel {
    pub fn from_rlda(m: &mut crate::rlda::RLDA) -> Self {
        Self {
            ldamodel: scalib::information::LDAModel::RLDA(m.inner.as_ref().unwrap().clone()))
        }
    }
}
pub(crate) struct ItSolver {
    inner: scalib::information::ItSolver,
}

#[pymethods]
impl ItSolver {
    #[new]
    fn new(py: Python, model) -> PyResult<Self> {
        
        
    }


}
