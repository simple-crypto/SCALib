extern crate ndarray;
mod belief_propagation;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, Array2, Axis};
use num_integer::binomial;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::{PyDict, PyList};

#[pymodule]
fn rust_stella(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "belief_propagation")]
    fn belief_propagation(
        _py: Python,
        functions: &PyList,
        variables: &PyList,
        it: usize,
        vertex: usize,
        nc: usize,
        n: usize,
    ) -> PyResult<()> {
        println!("vertex {:?}", vertex);
        let mut vertex: Vec<Array2<f64>> =
            (0..vertex).map(|_| Array2::<f64>::ones((n, nc))).collect();

        let pb = ProgressBar::new(functions.len() as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
        pb.set_message("Init functions...");
        let mut vec_funcs_id: Vec<(usize, usize)> = (0..vertex.len()).map(|_| (0, 0)).collect(); //(associated funct,position in fnc)

        let mut functions_rust: Vec<belief_propagation::Func> = functions
            .iter()
            .enumerate()
            .progress_with(pb)
            .map(|(i, x)| {
                let dict = x.downcast::<PyDict>().unwrap();
                let f = belief_propagation::to_func(dict);
                f.neighboors.iter().enumerate().for_each(|(j, x)| {
                    vec_funcs_id[*x] = (i, j);
                });
                f
            })
            .collect();

        let pb = ProgressBar::new(variables.len() as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
        pb.set_message("Init variables...");

        let mut vec_vars_id: Vec<(usize, usize)> = (0..vertex.len()).map(|_| (0, 0)).collect();
        let mut variables_rust: Vec<belief_propagation::Var> = variables
            .iter()
            .progress_with(pb)
            .enumerate()
            .map(|(i, x)| {
                let dict = x.downcast::<PyDict>().unwrap();
                let var = belief_propagation::to_var(dict);
                match &var.vartype {
                    belief_propagation::VarType::ProfilePara {
                        distri_orig,
                        distri_current: _,
                    } => var.neighboors.iter().enumerate().for_each(|(j, x)| {
                        let v = &mut vertex[*x];
                        v.assign(&distri_orig);
                        vec_vars_id[*x] = (i, j);
                    }),

                    belief_propagation::VarType::ProfileSingle {
                        distri_orig,
                        distri_current: _,
                    } => var.neighboors.iter().enumerate().for_each(|(j, x)| {
                        let v = &mut vertex[*x];
                        let distri = distri_orig.broadcast(v.shape()).unwrap();
                        v.assign(&distri);
                        vec_vars_id[*x] = (i, j);
                    }),
                    _ => var.neighboors.iter().enumerate().for_each(|(j, x)| {
                        vec_vars_id[*x] = (i, j);
                    }),
                }
                var
            })
            .collect();

        let pb = ProgressBar::new(it as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
    ));
        pb.set_message("Calculating BP...");

        for _ in (0..it).progress_with(pb) {
            unsafe{
                let mut vertex_for_func: Vec<Vec<&mut Array2<f64>>> = functions_rust.iter()
                    .map(|v| {let mut vec = Vec::<&mut Array2<f64>>::with_capacity(v.neighboors.len());
                                vec.set_len(v.neighboors.len());
                                vec
                    }).collect();
                vertex
                    .iter_mut()
                    .zip(vec_funcs_id.iter())
                    .for_each(|(x, (id, posi))| vertex_for_func[*id][*posi] = x);

                belief_propagation::update_functions(&mut functions_rust, &mut vertex_for_func);
            }

            unsafe{
                let mut vertex_for_var: Vec<Vec<&mut Array2<f64>>> = variables_rust.iter()
                    .map(|v| {let mut vec = Vec::<&mut Array2<f64>>::with_capacity(v.neighboors.len());
                                vec.set_len(v.neighboors.len());
                                vec
                    }).collect();
                vertex
                    .iter_mut()
                    .zip(vec_vars_id.iter())
                    .for_each(|(x, (id, posi))| vertex_for_var[*id][*posi] = x);
                belief_propagation::update_variables(&mut vertex_for_var,&mut variables_rust);
            }
        }
        let pb = ProgressBar::new(variables.len() as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
    ));
        pb.set_message("dump variables...");
        variables_rust
            .iter()
            .progress_with(pb)
            .zip(variables)
            .for_each(|(v_rust, v_python)| {
                let distri_current: &PyArray2<f64> =
                    v_python.get_item("distri").unwrap().extract().unwrap();
                let mut distri_current = unsafe { distri_current.as_array_mut() };
                match &v_rust.vartype {
                    belief_propagation::VarType::NotProfilePara {
                        distri_current: distri,
                    } => {
                        distri_current.assign(&distri);
                    }
                    belief_propagation::VarType::NotProfileSingle {
                        distri_current: distri,
                    } => {
                        distri_current.assign(&distri);
                    }
                    belief_propagation::VarType::ProfilePara {
                        distri_orig: _,
                        distri_current: distri,
                    } => {
                        distri_current.assign(&distri);
                    }
                    belief_propagation::VarType::ProfileSingle {
                        distri_orig: _,
                        distri_current: distri,
                    } => {
                        distri_current.assign(&distri);
                    }
                }
            });
        Ok(())
    }

    #[pyfn(m, "multivariate_pooled")]
    fn multivariate_pooled(
        _py: Python,
        u: PyReadonlyArray3<f64>, // U matrix (decomposition of Inv Cov (Npro x Npro)
        m: PyReadonlyArray2<f64>, // mean matrices (Nk x Npro)
        traces: PyReadonlyArray2<f64>, // the actual traces (N x Npro)
        prs: &PyArray2<f64>,      // the actual traces (N x Nk)
        det: PyReadonlyArray2<f64>, // (1,Nk)
    ) -> PyResult<()> {
        let u = u.as_array();
        let det = det.as_array();
        let traces = traces.as_array();
        let m = m.as_array();
        let mut prs = unsafe { prs.as_array_mut() };
        prs.axis_iter_mut(Axis(1)) // along Nk axis
            .into_par_iter()
            .zip(m.axis_iter(Axis(0)).into_par_iter())
            .zip(u.axis_iter(Axis(0)).into_par_iter())
            .zip(det.axis_iter(Axis(1)).into_par_iter())
            .for_each(|(((mut prs, m), u), det)| {
                let dev = &traces - &m;
                let tmp = dev
                    .dot(&u)
                    .mapv(|a| a.powi(2))
                    .sum_axis(Axis(1))
                    .mapv(|a| (-0.5 * a).exp())
                    / det[0];
                prs.assign(&tmp);
            });
        Ok(())
    }
    #[pyfn(m, "class_means_subs")]
    fn class_means_subs(
        _py: Python,
        labels: PyReadonlyArray1<u16>, // labels (N,)
        means: PyReadonlyArray2<f64>,  // the actual traces (N x Nk)
        traces_out: &PyArray2<f64>,    // where to store the results
    ) -> PyResult<()> {
        let mut traces_out = unsafe { traces_out.as_array_mut() };
        let labels = labels.as_array();
        let means = means.as_array();
        traces_out
            .axis_iter_mut(Axis(0)) // along Nk axis
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut traces_out)| {
                let x = labels[[i]] as usize;
                let m = means.slice(s![x, ..]);

                traces_out -= &m;
            });
        Ok(())
    }

    #[pyfn(m, "class_means_f64")]
    fn class_means_f64(
        _py: Python,
        u: PyReadonlyArray1<u16>,      // uniques labels
        labels: PyReadonlyArray1<u16>, // labels (N,)
        traces: PyReadonlyArray2<f64>, // the actual traces (N x Npro)
        means: &PyArray2<f64>,         // the actual traces (N x Nk)
    ) -> PyResult<()> {
        let u = u.as_array();
        let traces = traces.as_array();
        let labels = labels.as_array();
        let mut means = unsafe { means.as_array_mut() };
        u.axis_iter(Axis(0)) // along Nk axis
            .into_par_iter()
            .zip(means.axis_iter_mut(Axis(0)).into_par_iter())
            .for_each(|(u, mut mean)| {
                let mut n = 0;
                labels
                    .axis_iter(Axis(0))
                    .zip(traces.axis_iter(Axis(0)))
                    .for_each(|(lab, t)| {
                        if lab == u {
                            mean += &t.map(|x| (*x as f64));
                            n += 1;
                        }
                    });
                mean /= n as f64;
            });
        Ok(())
    }

    #[pyfn(m, "class_means")]
    fn class_means(
        _py: Python,
        u: PyReadonlyArray1<u16>,      // uniques labels
        labels: PyReadonlyArray1<u16>, // labels (N,)
        traces: PyReadonlyArray2<i16>, // the actual traces (N x Npro)
        means: &PyArray2<f64>,         // the actual traces (N x Nk)
    ) -> PyResult<()> {
        let u = u.as_array();
        let traces = traces.as_array();
        let labels = labels.as_array();
        let mut means = unsafe { means.as_array_mut() };
        u.axis_iter(Axis(0)) // along Nk axis
            .into_par_iter()
            .zip(means.axis_iter_mut(Axis(0)).into_par_iter())
            .for_each(|(u, mut mean)| {
                let mut n = 0;
                labels
                    .axis_iter(Axis(0))
                    .zip(traces.axis_iter(Axis(0)))
                    .for_each(|(lab, t)| {
                        if lab == u {
                            mean += &t.map(|x| (*x as f64));
                            n += 1;
                        }
                    });
                mean /= n as f64;
            });
        Ok(())
    }

    #[pyfn(m, "update_snrorder")]
    fn update_snrorder(
        _py: Python,
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        c: PyReadonlyArray2<u16>,      // (Np,len)
        n: &PyArray2<f64>,             // (Np,len)
        cs: &PyArrayDyn<f64>,          // (Np,Nc,D*2,N_sample)
        m: &PyArray3<f64>,             // (Np,Nc,N_sample)
        d: i32,
        nchunks: i32, //
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let c = c.as_array();
        let mut n = unsafe { n.as_array_mut() };
        let mut cs = unsafe { cs.as_array_mut() };
        let mut m = unsafe { m.as_array_mut() };
        let chunk_size = (traces.shape()[1] as i32 / nchunks) as usize;
        c.axis_iter(Axis(0))
            .into_par_iter()
            .zip(n.outer_iter_mut().into_par_iter())
            .zip(cs.outer_iter_mut().into_par_iter())
            .zip(m.outer_iter_mut().into_par_iter())
            .for_each(|(((c, mut n), mut cs), mut m)| {
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
                                .zip(traces.to_slice().unwrap().iter())
                                .zip(m.slice(s![x, ..]).to_slice().unwrap().iter())
                                .for_each(|((d, t), m)| *d = ((*t as f64) - (*m as f64)) / nx);
                            for j in (2..((d * 2) + 1)).rev() {
                                if nx > 1.0 {
                                    let r = cs.slice_mut(s![x, j - 1, ..]);
                                    let mult = (nx - 1.0).powi(j)
                                        * (1.0 - (-1.0 / (nx - 1.0)).powi(j - 1));
                                    r.into_slice()
                                        .unwrap()
                                        .iter_mut()
                                        .zip(delta.view().to_slice().unwrap().iter())
                                        .for_each(|(r, x)| {
                                            *r += x.powi(j as i32) * mult;
                                        });
                                }
                                for k in 1..((j - 2) + 1) {
                                    let i = (j - k - 1)..(j);
                                    let tab = cs.slice_mut(s![x, i;k, ..]);
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
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        c: PyReadonlyArray1<u8>,       // (len)
        n: &PyArray1<f64>,             // (len)
        cs: &PyArray3<f64>,            // (2,D*2,N_sample)
        m: &PyArray2<f64>,             // (2,N_sample)
        d: i32,
        nchunks: i32, //
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let c = c.as_array();
        let mut n = unsafe { n.as_array_mut() };
        let mut cs = unsafe { cs.as_array_mut() };
        let mut m = unsafe { m.as_array_mut() };
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
                        .zip(traces.to_slice().unwrap().iter())
                        .zip(m.slice(s![x, ..]).to_slice().unwrap().iter())
                        .for_each(|((d, t), m)| *d = ((*t as f64) - (*m as f64)) / nx);
                    for j in (2..((d * 2) + 1)).rev() {
                        if nx > 1.0 {
                            let r = cs.slice_mut(s![x, j - 1, ..]);
                            let mult = (nx - 1.0).powi(j) * (1.0 - (-1.0 / (nx - 1.0)).powi(j - 1));
                            r.into_slice()
                                .unwrap()
                                .iter_mut()
                                .zip(delta.view().to_slice().unwrap().iter())
                                .for_each(|(r, x)| {
                                    *r += x.powi(j as i32) * mult;
                                });
                        }
                        for k in 1..((j - 2) + 1) {
                            let i = (j - k - 1)..(j);
                            let tab = cs.slice_mut(s![x, i;k, ..]);
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
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        x: PyReadonlyArray2<u16>,      // (Np,len)
        sum: &PyArray3<i64>,           // (Np,Nc,N_sample)
        sum2: &PyArray3<i64>,          // (Np,Nc,N_sample)
        ns: &PyArray2<u32>,            // (Np,Nc)
        means: &PyArray3<f32>,         // (Np,Nc,N_sample)
        vars: &PyArray3<f32>,          // (Np,Nc,N_sample)
        snr: &PyArray2<f32>,           // (Np,N_sample)
        nchunks: i32,
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let x = x.as_array();
        let mut sum = unsafe { sum.as_array_mut() };
        let mut means = unsafe { means.as_array_mut() };
        let mut vars = unsafe { vars.as_array_mut() };
        let mut sum2 = unsafe { sum2.as_array_mut() };
        let mut ns = unsafe { ns.as_array_mut() };
        let mut snr = unsafe { snr.as_array_mut() };
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
                            for v in 0..nc {
                                let m = sum.slice_mut(s![v, ..]).into_slice().unwrap();
                                let sq = sum2.slice_mut(s![v, ..]).into_slice().unwrap();

                                for i in 0..n_traces {
                                    if v == x[[p, i]] as usize {
                                        let l = traces.slice(s![i, ..]);
                                        inner_loop_snr(m, sq, l.to_slice().unwrap());
                                    }
                                }
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
                                let m = means.slice_mut(s![i as usize, ..]).into_slice().unwrap();
                                let v = vars.slice_mut(s![i, ..]).into_slice().unwrap();

                                let s = sum.slice(s![i, ..]).to_slice().unwrap();
                                let s2 = sum2.slice(s![i, ..]).to_slice().unwrap();
                                let n = ns[i] as f32;
                                m.iter_mut()
                                    .zip(v.iter_mut())
                                    .zip(s.iter())
                                    .zip(s2.iter())
                                    .for_each(|(((m, v), s), s2)| {
                                        *m = (*s as f32) / n;
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
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        g: PyReadonlyArray2<u16>,      // (Ng,len)
        sumx: &PyArray2<f64>,          // (Ng,N_sample)
        sumx2: &PyArray2<f64>,         // (Ng,N_sample)
        sumxy: &PyArray2<f64>,         // (Ng,N_sample)
        sumy: &PyArray2<f64>,          // (Ng,N_sample)
        sumy2: &PyArray2<f64>,         // (Ng,N_sample)

        sm: PyReadonlyArray2<f64>, // (Nk,len)
        u: PyReadonlyArray2<f64>,  // (Nk,len)
        s: PyReadonlyArray2<f64>,  // (Nk,len)
        d: i32,
        nchunks: i32,
    ) -> PyResult<()> {
        let traces = traces.as_array();
        let g = g.as_array();
        let mut sumx = unsafe { sumx.as_array_mut() };
        let mut sumx2 = unsafe { sumx2.as_array_mut() };
        let mut sumxy = unsafe { sumxy.as_array_mut() };
        let mut sumy = unsafe { sumy.as_array_mut() };
        let mut sumy2 = unsafe { sumy2.as_array_mut() };

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
                                        sm_tmp.to_slice().unwrap(),
                                        s_tmp.to_slice().unwrap(),
                                        u_tmp.to_slice().unwrap(),
                                        l.to_slice().unwrap(),
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
/*fn inner_loop_class_means(m: &mut [f64], l: &[i16]) {
    m.iter_mut().zip(l.iter()).for_each(|(m, tr)| {
        *m += *tr as f64;
    });
}*/
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
