use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn str2method(s: &str) -> Result<ranklib::RankingMethod, &str> {
    match s {
        "naive" => Ok(ranklib::RankingMethod::Naive),
        "hist" => Ok(ranklib::RankingMethod::Hist),
        #[cfg(feature = "ntl")]
        "histbignum" => Ok(ranklib::RankingMethod::HistBigNum),
        #[cfg(feature = "hellib")]
        "hellib" => Ok(ranklib::RankingMethod::Hellib),
        _ => Err("Invalid method name"),
    }
}

#[pymodule]
fn ranklib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    #[pyfn(m, "rank_accuracy")]
    fn rank_accuracy(
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        acc: f64,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        let res = str2method(&method);
        match res {
            Ok(res) => {
                let res = res.rank_accuracy(&costs, &key, acc, merge);
                match res {
                    Ok(res) => Ok((res.min, res.est, res.max)),
                    Err(s) => {
                        println!("{}", s);
                        panic!()
                    }
                }
            }
            Err(_) => panic!(),
        }
        //return Ok((res.min, res.est, res.max));
    }

    #[pyfn(m, "rank_nbin")]
    fn rank_nbin(
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        nb_bin: usize,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        let res = str2method(&method);
        match res {
            Ok(res) => {
                let res = res.rank_nbin(&costs, &key, nb_bin, merge);
                match res {
                    Ok(res) => Ok((res.min, res.est, res.max)),
                    Err(s) => {
                        println!("{}", s);
                        panic!()
                    }
                }
            }
            Err(_) => panic!(),
        }
        //return Ok((res.min, res.est, res.max));
    }

    Ok(())
}
