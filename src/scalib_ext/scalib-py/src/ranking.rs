use pyo3::prelude::{pyfunction, PyResult, Python};

#[pyfunction]
#[pyo3(signature = (costs, key, acc, merge, method, max_nb_bin, config))]
pub fn rank_accuracy(
    py: Python,
    costs: Vec<Vec<f64>>,
    key: Vec<usize>,
    acc: f64,
    merge: Option<usize>,
    method: String,
    max_nb_bin: usize,
    config: crate::ConfigWrapper,
) -> PyResult<(f64, f64, f64)> {
    config.on_worker(py, |_| {
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
#[pyo3(signature = (costs, key, nb_bin, merge, method, config))]
pub fn rank_nbin(
    py: Python,
    costs: Vec<Vec<f64>>,
    key: Vec<usize>,
    nb_bin: usize,
    merge: Option<usize>,
    method: String,
    config: crate::ConfigWrapper,
) -> PyResult<(f64, f64, f64)> {
    config.on_worker(py, |_| {
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
