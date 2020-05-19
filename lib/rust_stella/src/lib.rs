extern crate ndarray;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, ArrayBase, Axis};
use num_integer::binomial;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rayon::prelude::*;
#[pymodule]
fn rust_stella(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "update_snrorder")]
    fn update_snrorder(
        _py: Python,
        traces: &PyArray2<i16>,   // (len,N_sample)
        c: &PyArray2<u16>,        // (Np,len)
        n: &mut PyArray2<f64>,    // (Np,len)
        cs: &mut PyArrayDyn<f64>, // (Np,Nc,D*2,N_sample)
        m: &mut PyArray3<f64>,    // (Np,Nc,N_sample)
        d: i32,
        nchunks: i32, //
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let c = c.as_array();
        let mut n = n.as_array_mut();
        let mut cs = cs.as_array_mut();
        let mut m = m.as_array_mut();
        let np = c.shape()[0];
        let chunk_size = (traces.shape()[1] as i32 / nchunks) as usize;
        c.axis_iter(Axis(0))
            .into_par_iter()
            .zip(n.outer_iter_mut().into_par_iter())
            .zip(cs.outer_iter_mut().into_par_iter())
            .zip(m.outer_iter_mut().into_par_iter())
            .enumerate()
            .for_each(|(p, (((c, mut n), mut cs), mut m))| {
                traces
                    .axis_chunks_iter(Axis(1), chunk_size)
                    .into_par_iter()
                    .zip(cs.axis_chunks_iter_mut(Axis(2), chunk_size).into_par_iter())
                    .zip(m.axis_chunks_iter_mut(Axis(1), chunk_size).into_par_iter())
                    .for_each(|((traces, mut cs), mut m)| {
                        let mut n = n.to_owned();
                        let mut delta = Array::<f64, _>::zeros(traces.shape()[1]);
                        traces.outer_iter().enumerate().for_each(|(i, traces)| {
                            // iterates over all the traces
                            let x = c[i] as usize;
                            n[[x]] += 1.0;
                            let nx = n[[x]];
                            delta
                                .view_mut()
                                .into_slice()
                                .unwrap()
                                .iter_mut()
                                .zip(traces.into_slice().unwrap().iter())
                                .zip(m.slice(s![x, ..]).into_slice().unwrap().iter())
                                .for_each({ |((d, t), m)| *d = ((*t as f64) - (*m as f64)) / nx });
                            for j in (2..((d * 2) + 1)).rev() {
                                if nx > 1.0 {
                                    let mut r = cs.slice_mut(s![x, j - 1, ..]);
                                    let mult = (nx - 1.0).powi(j)
                                        * (1.0 - (-1.0 / (nx - 1.0)).powi(j - 1));
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
                        });
                    });

                for i in 0..traces.shape()[0] {
                    let x = c[i] as usize;
                    n[[x]] += 1.0;
                }
            });
        Ok(())
    }

    #[pyfn(m, "update_ttest")]
    fn update_ttest(
        _py: Python,
        traces: &PyArray2<i16>, // (len,N_sample)
        c: &PyArray1<u8>,       // (len)
        n: &mut PyArray1<f64>,  // (len)
        cs: &mut PyArray3<f64>, // (2,D*2,N_sample)
        m: &mut PyArray2<f64>,  // (2,N_sample)
        d: i32,
        nchunks: i32, //
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let c = c.as_array();
        let mut n = n.as_array_mut();
        let mut cs = cs.as_array_mut();
        let mut m = m.as_array_mut();
        let chunk_size = (traces.shape()[1] as i32 / nchunks) as usize;
        traces
            .axis_chunks_iter(Axis(1), chunk_size)
            .into_par_iter()
            .zip(cs.axis_chunks_iter_mut(Axis(2), chunk_size).into_par_iter())
            .zip(m.axis_chunks_iter_mut(Axis(1), chunk_size).into_par_iter())
            .for_each(|((traces, mut cs), mut m)| {
                let mut n = n.to_owned();
                let mut delta = Array::<f64, _>::zeros(traces.shape()[1]);
                traces.outer_iter().enumerate().for_each(|(i, traces)| {
                    // iterates over all the traces
                    let x = c[i] as usize;
                    n[[x]] += 1.0;
                    let nx = n[[x]];
                    delta
                        .view_mut()
                        .into_slice()
                        .unwrap()
                        .iter_mut()
                        .zip(traces.into_slice().unwrap().iter())
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
                });
            });
        for i in 0..traces.shape()[0] {
            let x = c[i] as usize;
            n[[x]] += 1.0;
        }
        Ok(())
    }
    #[pyfn(m, "update_snr")]
    fn update_snr(
        _py: Python,
        traces: &PyArray2<i16>,    // (len,N_sample)
        x: &PyArray2<u16>,         // (Np,len)
        sum: &mut PyArray3<i64>,   // (Np,Nc,N_sample)
        sum2: &mut PyArray3<i64>,  // (Np,Nc,N_sample)
        ns: &mut PyArray2<u32>,    // (Np,Nc)
        means: &mut PyArray3<f32>, // (Np,Nc,N_sample)
        vars: &mut PyArray3<f32>,  // (Np,Nc,N_sample)
        snr: &mut PyArray2<f32>,   // (Np,N_sample)
        nchunks: i32,
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let x = x.as_array();
        let mut sum = sum.as_array_mut();
        let mut means = means.as_array_mut();
        let mut vars = vars.as_array_mut();
        let mut sum2 = sum2.as_array_mut();
        let mut ns = ns.as_array_mut();
        let mut snr = snr.as_array_mut();
        let n_traces = traces.shape()[0];
        let nc = sum.shape()[1];
        let chunk_size = (traces.shape()[1] as i32 / nchunks) as usize;
        sum.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(sum2.outer_iter_mut().into_par_iter())
            .zip(ns.outer_iter_mut().into_par_iter())
            .zip(means.outer_iter_mut().into_par_iter())
            .zip(vars.outer_iter_mut().into_par_iter())
            .zip(snr.outer_iter_mut().into_par_iter())
            .enumerate()
            .for_each(
                |(p, (((((mut sum, mut sum2), mut ns), mut means), mut vars), mut snr))| {
                    traces
                        .axis_chunks_iter(Axis(1), chunk_size)
                        .into_par_iter()
                        .zip(
                            sum.axis_chunks_iter_mut(Axis(1), chunk_size)
                                .into_par_iter(),
                        )
                        .zip(
                            sum2.axis_chunks_iter_mut(Axis(1), chunk_size)
                                .into_par_iter(),
                        )
                        .for_each(|((traces, mut sum), mut sum2)| {
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
                            }
                        });
                    for i in 0..n_traces {
                        let v = x[[p, i]] as usize;
                        ns[v] += 1;
                    }
                    means
                        .axis_chunks_iter_mut(Axis(1), chunk_size)
                        .into_par_iter()
                        .zip(
                            vars.axis_chunks_iter_mut(Axis(1), chunk_size)
                                .into_par_iter(),
                        )
                        .zip(sum.axis_chunks_iter(Axis(1), chunk_size).into_par_iter())
                        .zip(sum2.axis_chunks_iter(Axis(1), chunk_size).into_par_iter())
                        .zip(
                            snr.axis_chunks_iter_mut(Axis(0), chunk_size)
                                .into_par_iter(),
                        )
                        .for_each(|((((mut means, mut vars), sum), sum2), mut snr)| {
                            for i in 0..nc {
                                let mut m =
                                    means.slice_mut(s![(i as usize), ..]).into_slice().unwrap();
                                let mut v = vars.slice_mut(s![i, ..]).into_slice().unwrap();

                                let s = sum.slice(s![i, ..]).into_slice().unwrap();
                                let s2 = sum2.slice(s![i, ..]).into_slice().unwrap();
                                let n = ns[i] as f32;
                                m.iter_mut()
                                    .zip(v.iter_mut())
                                    .zip(s.iter())
                                    .zip(s2.iter())
                                    .for_each(|(((m, v), s), s2)| {
                                        *m = ((*s as f32) / n);
                                        let tmp = *m;
                                        *v = ((*s2 as f32) / n) - tmp.powi(2);
                                    });
                            }
                            let num = means.var_axis(Axis(0), 1.0);
                            let den = vars.mean_axis(Axis(0)).unwrap();
                            //#let x = means + vars;
                            let tmp = num / den;
                            snr.assign(&tmp);
                        });
                },
            );

        Ok(())
    }

    #[pyfn(m, "update_mcp_dpa")]
    fn update_mcp_dpa(
        _py: Python,
        traces: &PyArray2<i16>,    // (len,N_sample)
        g: &PyArray2<u16>,         // (Ng,len)
        sumx: &mut PyArray2<f64>,  // (Ng,N_sample)
        sumx2: &mut PyArray2<f64>, // (Ng,N_sample)
        sumxy: &mut PyArray2<f64>, // (Ng,N_sample)
        sumy: &mut PyArray2<f64>,  // (Ng,N_sample)
        sumy2: &mut PyArray2<f64>, // (Ng,N_sample)

        sm: &PyArray2<f64>, // (Nk,len)
        u: &PyArray2<f64>,  // (Nk,len)
        s: &PyArray2<f64>,  // (Nk,len)
        d: i32,
        nchunks: i32,
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let g = g.as_array();
        let mut sumx = sumx.as_array_mut();
        let mut sumx2 = sumx2.as_array_mut();
        let mut sumxy = sumxy.as_array_mut();
        let mut sumy = sumy.as_array_mut();
        let mut sumy2 = sumy2.as_array_mut();

        let sm = sm.as_array();
        let u = u.as_array();
        let s = s.as_array();
        let n_traces = traces.shape()[0];
        let chunk_size = (traces.shape()[1] as i32 / nchunks) as usize;
        g.axis_iter(Axis(0))
            .into_par_iter()
            .zip(sumx.outer_iter_mut().into_par_iter())
            .zip(sumx2.outer_iter_mut().into_par_iter())
            .zip(sumxy.outer_iter_mut().into_par_iter())
            .zip(sumy.outer_iter_mut().into_par_iter())
            .zip(sumy2.outer_iter_mut().into_par_iter())
            .for_each(
                |(((((g, mut sumx), mut sumx2), mut sumxy), mut sumy), mut sumy2)| {
                    traces
                        .axis_chunks_iter(Axis(1), chunk_size)
                        .into_par_iter()
                        .zip(
                            sumx.axis_chunks_iter_mut(Axis(0), chunk_size)
                                .into_par_iter(),
                        )
                        .zip(
                            sumx2
                                .axis_chunks_iter_mut(Axis(0), chunk_size)
                                .into_par_iter(),
                        )
                        .zip(
                            sumxy
                                .axis_chunks_iter_mut(Axis(0), chunk_size)
                                .into_par_iter(),
                        )
                        .zip(
                            sumy.axis_chunks_iter_mut(Axis(0), chunk_size)
                                .into_par_iter(),
                        )
                        .zip(
                            sumy2
                                .axis_chunks_iter_mut(Axis(0), chunk_size)
                                .into_par_iter(),
                        )
                        .for_each(
                            |(
                                ((((traces, mut sumx), mut sumx2), mut sumxy), mut sumy),
                                mut sumy2,
                            )| {
                                for i in 0..n_traces {
                                    let v = g[[i]] as usize;
                                    let sm_tmp = sm.slice(s![v, ..]);
                                    let s_tmp = s.slice(s![v, ..]);
                                    let u_tmp = u.slice(s![v, ..]);
                                    let l = traces.slice(s![i, ..]);
                                    inner_loop_mcp_dpa(
                                        sumx.as_slice_mut().unwrap(),
                                        sumx2.as_slice_mut().unwrap(),
                                        sumy.as_slice_mut().unwrap(),
                                        sumy2.as_slice_mut().unwrap(),
                                        sumxy.as_slice_mut().unwrap(),
                                        sm_tmp.into_slice().unwrap(),
                                        s_tmp.into_slice().unwrap(),
                                        u_tmp.into_slice().unwrap(),
                                        l.into_slice().unwrap(),
                                        d,
                                    );
                                }
                            },
                        );
                },
            );

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
fn inner_loop_mcp_dpa(
    sumx: &mut [f64],
    sumx2: &mut [f64],
    sumy: &mut [f64],
    sumy2: &mut [f64],
    sumxy: &mut [f64],
    sm: &[f64],
    s: &[f64],
    u: &[f64],
    l: &[i16],
    d: i32,
) {
    sumx.iter_mut()
        .zip(sumx2.iter_mut())
        .zip(sumy.iter_mut())
        .zip(sumy2.iter_mut())
        .zip(sumxy.iter_mut())
        .zip(sm.iter())
        .zip(s.iter())
        .zip(u.iter())
        .zip(l.iter())
        .for_each(
            |((((((((sumx, sumx2), sumy), sumy2), sumxy), sm), s), u), tr)| {
                let x = (((*tr as f64) - u) / s).powi(d);
                let y = sm;
                *sumx += x;
                *sumx2 += x * x;
                *sumxy += x * y;
                *sumy2 += y * y;
                *sumy += y;
            },
        );
}

fn inner_loop_ttest(dest: &mut [f64], cs: &[f64], delta: &[f64], comb: f64, k: i32) {
    dest.iter_mut()
        .zip(delta.iter())
        .zip(cs.iter())
        .for_each(|((dest, delta), cs)| *dest += comb * (-delta).powi(k) * cs);
}
