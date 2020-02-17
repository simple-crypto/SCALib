use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

#[pymodule]
fn rust_stella(_py: Python, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py(
        py: Python,
        a: f64,
        x: &PyArrayDyn<f64>,
        y: &PyArrayDyn<f64>,
    ) -> Py<PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py).to_owned()
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut();
        mult(a, x);
        Ok(())
    }

    #[pyfn(m, "update_snr")]
    fn update_snr(
        _py: Python,
        traces: &PyArrayDyn<i16>,
        x: &PyArrayDyn<u16>,
        sum: &mut PyArrayDyn<i64>,
        sum2: &mut PyArrayDyn<i64>,
        ns: &mut PyArrayDyn<u32>,
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let x = x.as_array();
        let mut sum = sum.as_array_mut();
        let mut sum2 = sum2.as_array_mut();
        let mut ns = ns.as_array_mut();

        let n_traces = traces.shape()[0];
        let len = traces.shape()[1];
        let np = sum.shape()[0];
        for i in 0..n_traces {
            for t in 0..len {
                let l = traces[[i, t]] as i32;
                let l2 = l * l;
                for p in 0..np {
                    let v = x[[p, i]] as usize;
                    sum[[p, v, t]] += l as i64;
                    sum2[[p, v, t]] += l2 as i64;
                }
            }
            for p in 0..np {
                let v = x[[p, i]] as usize;
                ns[[p, v]] += 1;
            }
        }
        Ok(())
    }

    #[pyfn(m, "my_py")]
    fn my_py(_py: Python, x: &mut PyArrayDyn<f64>) -> PyResult<()> {
        let mut x = x.as_array_mut();
        let s = x.shape();
        for i in 0..s[0] {
            x[i] = (i as f64) * 2.0;
        }
        Ok(())
    }
    Ok(())
}
