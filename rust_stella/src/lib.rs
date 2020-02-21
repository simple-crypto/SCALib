extern crate ndarray;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, ArrayBase, Axis};
use num_integer::binomial;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rayon::prelude::*;
#[pymodule]
fn rust_stella(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "update_ttest")]
    fn update_ttest(
        _py: Python,
        traces: &PyArray2<i16>, // (len,N_sample)
        c: &PyArray1<u16>,      // (len)
        n: &mut PyArray1<f64>,  // (len)
        cs: &mut PyArray3<f64>, // (2,D*2,N_sample)
        m: &mut PyArray2<f64>,  // (2,D*2,N_sample)
        d: i32,                 //
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let c = c.as_array();
        let mut n = n.as_array_mut();
        let mut cs = cs.as_array_mut();
        let mut m = m.as_array_mut();
        let n_traces = traces.shape()[0];
        let mut delta = Array::<f64, _>::zeros(traces.shape()[1]);

        for i in 0..n_traces {
            let x = c[i] as usize;
            n[[x]] += 1.0;
            let nx = n[[x]];
            delta
                .view_mut()
                .into_slice()
                .unwrap()
                .iter_mut()
                .zip(traces.slice(s![i, ..]).into_slice().unwrap().iter())
                .zip(m.slice(s![x, ..]).into_slice().unwrap().iter())
                .for_each({ |((d, t), m)| *d = ((*t as f64) - (*m as f64)) / nx });
            for j in (2..((d * 2) + 1)).rev() {
                if nx > 1.0 {
                    let mut r = cs.slice_mut(s![x, j - 1, ..]);
                    let mult = (nx - 1.0).powi(j) * (1.0 - (-1.0 / (nx - 1.0)).powi(j - 1));
                    r.into_slice()
                        .unwrap()
                        .iter_mut()
                        .zip(delta.view().into_slice().unwrap().iter())
                        .for_each(|(r, x)| {
                            *r += x.powi(j as i32) * mult;
                        });
                }
                for k in 1..((j - 2) + 1) {
                    let I = ((j - k - 1)..(j));
                    let tab = cs.slice_mut(s![x, I;k, ..]);
                    let (a, b) = tab.split_at(Axis(0), 1);
                    let cb = binomial(j, k) as f64;
                    inner_loop_ttest(
                        b.into_slice().unwrap(),
                        a.into_slice().unwrap(),
                        delta.as_slice().unwrap(),
                        cb,
                        k,
                    );
                }
            }
            let mut ret = m.slice_mut(s![x, ..]);
            ret += &(delta);
            cs.slice_mut(s![x, 0, ..]).assign(&ret);
        }
        Ok(())
    }

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
                    let m = sum.slice_mut(s![v, ..]);
                    let sq = sum2.slice_mut(s![v, ..]);
                    let l = traces.slice(s![i, ..]);
                    inner_loop_snr(
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
fn inner_loop_snr(m: &mut [i64], sq: &mut [i64], l: &[i16]) {
    m.iter_mut()
        .zip(sq.iter_mut())
        .zip(l.iter())
        .for_each(|((m, sq), tr)| {
            *m += *tr as i64;
            *sq += (*tr as i64) * (*tr as i64);
        });
}

fn inner_loop_ttest(dest: &mut [f64], cs: &[f64], delta: &[f64], comb: f64, k: i32) {
    dest.iter_mut()
        .zip(delta.iter())
        .zip(cs.iter())
        .for_each(|((dest, delta), cs)| *dest += comb * (-delta).powi(k) * cs);
}
