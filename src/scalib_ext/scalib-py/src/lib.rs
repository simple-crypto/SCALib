use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

mod belief_propagation;
mod lda;
mod ranking;
mod snr;
mod ttest;

#[pyfunction]
pub fn run_bp(
    py: Python,
    functions: &PyList,
    variables: &PyList,
    it: usize,
    vertex: usize,
    nc: usize,
    n: usize,
    progress: bool,
) -> PyResult<()> {
    belief_propagation::run_bp(py, functions, variables, it, vertex, nc, n, progress)
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

#[pymodule]
fn _scalib_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<snr::SNR>()?;
    m.add_class::<ttest::Ttest>()?;
    m.add_class::<ttest::MTtest>()?;
    m.add_class::<lda::LDA>()?;
    m.add_class::<lda::LdaAcc>()?;
    m.add_function(wrap_pyfunction!(ranking::rank_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(ranking::rank_nbin, m)?)?;
    m.add_function(wrap_pyfunction!(run_bp, m)?)?;
    m.add_function(wrap_pyfunction!(partial_cp, m)?)?;

    Ok(())
}
