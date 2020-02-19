use ndarray::s;
use numpy::{PyArray2, PyArray3};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
#[pymodule]
fn rust_stella(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "update_snr")]
    fn update_snr(
        _py: Python,
        traces: &PyArray2<i16>,   // (len,N_sample)
        x: &PyArray2<u16>,        // (Np,len)
        sum: &mut PyArray3<i64>,  // (Np,Nc,N_sample)
        sum2: &mut PyArray3<i64>, // (Np,Nc,N_sample)
        ns: &mut PyArray2<u32>,   // (Np,Nc)
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let x = x.as_array();
        let mut sum = sum.as_array_mut();
        let mut sum2 = sum2.as_array_mut();
        let mut ns = ns.as_array_mut();

        let n_traces = traces.shape()[0];
        let np = sum.shape()[0] as usize;

        for p in 0..np {
            for i in 0..n_traces {
                let v = x[[p, i]] as usize;
                let mut m = sum.slice_mut(s![p, v, ..]);
                let mut sq = sum2.slice_mut(s![p, v, ..]);
                let l = traces.slice(s![i, ..]);
                hello_gae(
                    m.into_slice().unwrap(),
                    sq.into_slice().unwrap(),
                    l.into_slice().unwrap(),
                );
                ns[[p, v]] += 1;
            }
        }

        Ok(())
    }

    Ok(())
}
#[inline(never)]
fn hello_gae(mut m: &mut [i64], mut sq: &mut [i64], l: &[i16]) {
    m.iter_mut()
        .zip(sq.iter_mut())
        .zip(l.iter())
        .for_each(|((m, sq), tr)| {
            *m += *tr as i64;
            *sq += (*tr as i64) * (*tr as i64);
        });
}
