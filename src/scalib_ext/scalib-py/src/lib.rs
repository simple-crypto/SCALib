// Require this to link openblas_src in the final binary, as otherwise it appears to be not used
// and cargo does not link it.
// This seems to be a strange combination of not using explicitly functions declared by
// openblas_src (as they are declared by blas-sys), and not requiring those symbols from a
// "top-level" rust function (due to top-level re-writing of #[pymodule]).
// To be reported/investigated.
// Not required on windows, as we directly link openblas there from build.rs.
#[cfg(not(windows))]
extern crate openblas_src as _;

use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::PyList;
use scalib::lda;
use scalib::snr;
use scalib::ttest;

fn str2method(s: &str) -> Result<ranklib::RankingMethod, &str> {
    match s {
        "naive" => Ok(ranklib::RankingMethod::Naive),
        "hist" => Ok(ranklib::RankingMethod::Hist),
        #[cfg(feature = "ntl")]
        "histbignum" => Ok(ranklib::RankingMethod::HistBigNum),
        #[cfg(not(feature = "ntl"))]
        "histbignum" => Err("Ranking method 'histbignum' is not supported. Compile scalib_ext with ntl feature enabled."),
        #[cfg(feature = "hellib")]
        "hellib" => Ok(ranklib::RankingMethod::Hellib),
        #[cfg(not(feature = "hellib"))]
        "hellib" => Err("Ranking method 'hellib' is not supported. Compile scalib_ext with hellib feature enabled."),
        _ => Err(
            "Invalid ranking method name."
        ),
    }
}

#[pymodule]
fn _scalib_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<snr::SNR>()?;
    m.add_class::<ttest::Ttest>()?;
    m.add_class::<lda::LDA>()?;

    #[pyfn(m, "run_bp")]
    pub fn run_bp(
        py: Python,
        functions: &PyList,
        variables: &PyList,
        it: usize,
        vertex: usize,
        nc: usize,
        n: usize,
    ) -> PyResult<()> {
        scalib::belief_propagation::run_bp(py, functions, variables, it, vertex, nc, n)
    }

    #[pyfn(m, "partial_cp")]
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

    #[pyfn(m, "rank_accuracy")]
    fn rank_accuracy(
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        acc: f64,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        let res = str2method(&method).unwrap_or_else(|s| panic!("{}", s));
        let res = res.rank_accuracy(&costs, &key, acc, merge);
        match res {
            Ok(res) => Ok((res.min, res.est, res.max)),
            Err(s) => {
                panic!("{}", s);
            }
        }
    }

    #[pyfn(m, "rank_nbin")]
    fn rank_nbin(
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        nb_bin: usize,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        let res = str2method(&method).unwrap_or_else(|s| panic!("{}", s));
        let res = res.rank_nbin(&costs, &key, nb_bin, merge);
        match res {
            Ok(res) => Ok((res.min, res.est, res.max)),
            Err(s) => {
                panic!("{}", s);
            }
        }
    }

    Ok(())
}
