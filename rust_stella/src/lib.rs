extern crate ndarray;
use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
use numpy::{PyArray2, PyArray3};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rayon::prelude::*;

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
        sum.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(sum2.outer_iter_mut().into_par_iter())
            .zip(ns.outer_iter_mut().into_par_iter())
            .enumerate()
            .for_each(|(p, ((mut sum, mut sum2), mut ns))| {
                for i in 0..n_traces {
                    let v = x[[p, i]] as usize;
                    let mut m = sum.slice_mut(s![v, ..]);
                    let mut sq = sum2.slice_mut(s![v, ..]);
                    let l = traces.slice(s![i, ..]);
                    inter_loop(
                        m.into_slice().unwrap(),
                        sq.into_slice().unwrap(),
                        l.into_slice().unwrap(),
                    );
                    ns[v] += 1;
                }
            });

        Ok(())
    }

    Ok(())
}
fn inter_loop(mut m: &mut [i64], mut sq: &mut [i64], l: &[i16]) {
    m.iter_mut()
        .zip(sq.iter_mut())
        .zip(l.iter())
        .for_each(|((m, sq), tr)| {
            *m += *tr as i64;
            *sq += (*tr as i64) * (*tr as i64);
        });
}
