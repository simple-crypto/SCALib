use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::error::Error;

mod belief_propagation;
mod factor_graph;
mod lda;
mod ranking;
mod snr;
mod thread_pool;
mod ttest;

create_exception!(_scalib_ext, ScalibError, PyException);

impl ScalibError {
    fn from_scalib(x: scalib::ScalibError, py: Python<'_>) -> PyErr {
        let mut e = ScalibError::new_err(x.to_string());
        annotate_cause(x.source(), &mut e, py);
        e
    }
}

fn annotate_cause(err: Option<&(dyn Error + 'static)>, pyerr: &mut PyErr, py: Python) {
    if let Some(e) = err {
        let mut sub_pyerr = ScalibError::new_err(e.to_string());
        annotate_cause(e.source(), &mut sub_pyerr, py);
        pyerr.set_cause(py, Some(sub_pyerr));
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Config {
    inner: scalib::Config,
}

#[pymethods]
impl Config {
    #[new]
    fn new() -> Self {
        Self {
            inner: Default::default(),
        }
    }
    fn show_progress(&mut self, show: bool) {
        self.inner.show_progress = show;
    }
    fn progress(&self) -> bool {
        self.inner.show_progress
    }
}

#[derive(FromPyObject)]
struct ThreadPoolWrapper<'p> {
    pool: PyRef<'p, thread_pool::ThreadPool>,
}

#[derive(FromPyObject)]
pub struct ConfigWrapper<'p> {
    #[pyo3(attribute("threadpool"))]
    threadpool: ThreadPoolWrapper<'p>,
    #[pyo3(attribute("inner"))]
    config: PyRef<'p, Config>,
}

impl<'p> ConfigWrapper<'p> {
    fn on_worker<OP, R>(&self, py: Python, op: OP) -> R
    where
        OP: FnOnce(&scalib::Config) -> R + Send,
        R: Send,
    {
        let pool = &self.threadpool.pool.pool;
        let cfg = &self.config.inner;
        py.allow_threads(|| pool.install(|| op(cfg)))
    }
}

#[pyfunction]
fn partial_cp(
    _py: Python,
    traces: PyReadonlyArray2<i16>, // (len,N_sample)
    poi: PyReadonlyArray1<u32>,    // (Np,len)
    store: &PyArray2<i16>,
) {
    let traces = traces.as_array();
    let poi = poi.as_array();
    let mut store = unsafe { store.as_array_mut() };
    store
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(poi.outer_iter().into_par_iter())
        .for_each(|(mut x, poi)| {
            let poi = poi.first().unwrap();
            x.assign(&traces.slice(s![.., *poi as usize]));
        });
}

#[pyfunction]
fn usable_parallelism(_py: Python) -> usize {
    // Concepturally, num_cpus::get() gives the available parallelism, like
    // std::thread::available_parallelism.
    // However, this counts hyper-threading, which is most likely that not to
    // harm SCALib's performance.
    // We therefore clip the parallelism to the number of physical cpus.
    std::cmp::min(num_cpus::get(), num_cpus::get_physical())
}

#[pymodule]
fn _scalib_ext(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("ScalibError", py.get_type::<ScalibError>())?;
    m.add_class::<Config>()?;
    m.add_class::<snr::SNR>()?;
    m.add_class::<ttest::Ttest>()?;
    m.add_class::<ttest::MTtest>()?;
    m.add_class::<lda::LDA>()?;
    m.add_class::<lda::LdaAcc>()?;
    m.add_class::<factor_graph::FactorGraph>()?;
    m.add_class::<factor_graph::BPState>()?;
    m.add_class::<thread_pool::ThreadPool>()?;
    m.add_function(wrap_pyfunction!(ranking::rank_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(ranking::rank_nbin, m)?)?;
    m.add_function(wrap_pyfunction!(belief_propagation::run_bp, m)?)?;
    m.add_function(wrap_pyfunction!(partial_cp, m)?)?;
    m.add_function(wrap_pyfunction!(usable_parallelism, m)?)?;

    Ok(())
}
