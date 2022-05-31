use crate::thread_pool::ThreadPool;
use pyo3::prelude::{pyfunction, PyResult, Python};

#[pyfunction]
pub fn rank_accuracy(
    py: Python,
    costs: Vec<Vec<f64>>,
    key: Vec<usize>,
    acc: f64,
    merge: Option<usize>,
    method: String,
    max_nb_bin: usize,
    thread_pool: &ThreadPool,
) -> PyResult<(f64, f64, f64)> {
    crate::on_worker(py, thread_pool, || {
        let res = str2method(&method).unwrap_or_else(|s| panic!("{}", s));
        let res = res.rank_accuracy(&costs, &key, acc, merge, max_nb_bin);
        match res {
            Ok(res) => Ok((res.min, res.est, res.max)),
            Err(s) => {
                panic!("{}", s);
            }
        }
    })
}

#[pyfunction]
pub fn rank_nbin(
    py: Python,
    costs: Vec<Vec<f64>>,
    key: Vec<usize>,
    nb_bin: usize,
    merge: Option<usize>,
    method: String,
    thread_pool: &ThreadPool,
) -> PyResult<(f64, f64, f64)> {
    crate::on_worker(py, thread_pool, || {
        let res = str2method(&method).unwrap_or_else(|s| panic!("{}", s));
        let res = res.rank_nbin(&costs, &key, nb_bin, merge);
        match res {
            Ok(res) => Ok((res.min, res.est, res.max)),
            Err(s) => {
                panic!("{}", s);
            }
        }
    })
}

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
