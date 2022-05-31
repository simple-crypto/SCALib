use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

mod belief_propagation;
mod lda;
mod ranking;
mod snr;
mod thread_pool;
mod ttest;

pub(crate) fn on_worker<OP, R>(py: Python, thread_pool: &thread_pool::ThreadPool, op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    py.allow_threads(|| thread_pool.pool.install(op))
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
fn get_n_cpus_physical(_py: Python) -> usize {
    num_cpus::get_physical()
}

#[pymodule]
fn _scalib_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<snr::SNR>()?;
    m.add_class::<ttest::Ttest>()?;
    m.add_class::<ttest::MTtest>()?;
    m.add_class::<lda::LDA>()?;
    m.add_class::<lda::LdaAcc>()?;
    m.add_class::<thread_pool::ThreadPool>()?;
    m.add_function(wrap_pyfunction!(ranking::rank_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(ranking::rank_nbin, m)?)?;
    m.add_function(wrap_pyfunction!(belief_propagation::run_bp, m)?)?;
    m.add_function(wrap_pyfunction!(partial_cp, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_cpus_physical, m)?)?;

    Ok(())
}
