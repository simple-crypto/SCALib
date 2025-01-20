use pyo3::prelude::*;

#[pyfunction]
fn support_x86_64_v3(_py: Python) -> Vec<(&'static str, bool)> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        vec![
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("bmi1", std::is_x86_feature_detected!("bmi1")),
            ("bmi2", std::is_x86_feature_detected!("bmi2")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("lzcnt", std::is_x86_feature_detected!("lzcnt")),
            ("movbe", std::is_x86_feature_detected!("movbe")),
        ]
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        vec![]
    }
}

#[pymodule]
fn _cpu_check(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(support_x86_64_v3, m)?)?;
    Ok(())
}
