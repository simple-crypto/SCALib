use ndarray::s;
use numpy::PyArrayDyn;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn rust_stella(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "update_snr")]
    fn update_snr(
        _py: Python,
        traces: &PyArrayDyn<i16>,   // (len,N_sample)
        x: &PyArrayDyn<u16>,        // (Np,len)
        sum: &mut PyArrayDyn<i64>,  // (Np,Nc,N_sample)
        sum2: &mut PyArrayDyn<i64>, // (Np,Nc,N_sample)
        ns: &mut PyArrayDyn<u32>,   // (Np,Nc)
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let x = x.as_array();
        let mut sum = sum.as_array_mut();
        let mut sum2 = sum2.as_array_mut();
        let mut ns = ns.as_array_mut();

        let n_traces = traces.shape()[0];
        //let len = traces.shape()[1];
        let np = sum.shape()[0];
        for i in 0..n_traces {
            let l = traces.slice(s![i, ..]).mapv(|x| x as i64);
            let l2 = l.mapv(|x| x * x);
            for p in 0..np {
                let v = x[[p, i]] as usize;
                let mut sl = sum.slice_mut(s![p, v, ..]); //(N_sample)
                sl += &l.slice(s![..]);
                let mut sl = sum2.slice_mut(s![p, v, ..]); //(N_sample)
                sl += &l2.slice(s![..]);
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
