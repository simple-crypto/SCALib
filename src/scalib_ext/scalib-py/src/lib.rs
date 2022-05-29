use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::PyList;
mod belief_propagation;
mod lda;
mod snr;
mod ttest;

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

create_exception!(_scalib_ext, CPUInstructionError, PyException);

#[pymodule]
fn _scalib_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    #[cfg(target_feature = "avx2")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        let err = CPUInstructionError::new_err("This version of SCALib is compiled with AVX2 but the AVX2 feature is not detected for the current CPU. To run SCALib, re-compile  (see https://github.com/simple-crypto/SCALib/blob/main/DEVELOP.rst.)");
        return Err(err);
    }
    m.add("CPUInstructionError", _py.get_type::<CPUInstructionError>())?;
    m.add_class::<snr::SNR>()?;
    m.add_class::<ttest::Ttest>()?;
    m.add_class::<ttest::MTtest>()?;
    m.add_class::<lda::LDA>()?;
    m.add_class::<lda::LdaAcc>()?;

    #[pyfn(m, "run_bp")]
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

    #[pyfn(m, "rank_accuracy")]
    fn rank_accuracy(
        py: Python,
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        acc: f64,
        merge: Option<usize>,
        method: String,
        max_nb_bin: usize,
    ) -> PyResult<(f64, f64, f64)> {
        py.allow_threads(|| {
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

    #[pyfn(m, "rank_nbin")]
    fn rank_nbin(
        py: Python,
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        nb_bin: usize,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        py.allow_threads(|| {
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

    Ok(())
}
