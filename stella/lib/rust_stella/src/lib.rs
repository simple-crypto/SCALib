extern crate ndarray;
extern crate openblas_src;
mod belief_propagation;
mod lda;
mod snr;
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
        // array to save all the vertex
        let mut vertex: Vec<Array2<f64>> =
            (0..vertex).map(|_| Array2::<f64>::ones((n, nc))).collect();

        // mapping of the vertex for functions and variables
        let mut vec_funcs_id: Vec<(usize, usize)> = (0..vertex.len()).map(|_| (0, 0)).collect(); //(associated funct,position in fnc)
        let mut vec_vars_id: Vec<(usize, usize)> = (0..vertex.len()).map(|_| (0, 0)).collect();

        // loading bar
        let pb = ProgressBar::new(functions.len() as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
        pb.set_message("Init functions...");

        // map all python functions to rust ones + generate the mapping in vec_functs_id
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

        // loading bar
        let pb = ProgressBar::new(variables.len() as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
        pb.set_message("Init variables...");

        // map all python var to rust ones
        // generate the vertex mapping in vec_vars_id
        // init the messages along the edges with initial distributions
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

        // loading bar
        let pb = ProgressBar::new(it as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
    ));
        pb.set_message("Calculating BP...");

        for _ in (0..it).progress_with(pb) {
            unsafe {
                // map vertex to vec<vec<>> based on vec_funcs_id
                let mut vertex_for_func: Vec<Vec<&mut Array2<f64>>> = functions_rust
                    .iter()
                    .map(|v| {
                        let mut vec = Vec::<&mut Array2<f64>>::with_capacity(v.neighboors.len());
                        vec.set_len(v.neighboors.len());
                        vec
                    })
                    .collect();
                vertex
                    .iter_mut()
                    .zip(vec_funcs_id.iter())
                    .for_each(|(x, (id, posi))| vertex_for_func[*id][*posi] = x);

                // unpdate function nodes
                belief_propagation::update_functions(&mut functions_rust, &mut vertex_for_func);
            }

            unsafe {
                // map vertex to vec<vec<>> based on vec_vars_id
                let mut vertex_for_var: Vec<Vec<&mut Array2<f64>>> = variables_rust
                    .iter()
                    .map(|v| {
                        let mut vec = Vec::<&mut Array2<f64>>::with_capacity(v.neighboors.len());
                        vec.set_len(v.neighboors.len());
                        vec
                    })
                    .collect();
                vertex
                    .iter_mut()
                    .zip(vec_vars_id.iter())
                    .for_each(|(x, (id, posi))| vertex_for_var[*id][*posi] = x);

                // variables function nodes
                belief_propagation::update_variables(&mut vertex_for_var, &mut variables_rust);
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
    #[pyfn(m, "lda_matrix")]
    fn lda_matrix(
        _py: Python,
        x: PyReadonlyArray2<i16>, // U matrix (decomposition of Inv Cov (Npro x Npro)
        y: PyReadonlyArray1<u16>, // mean matrices (Nk x Npro)
        sb: &PyArray2<f64>,       // the actual traces (N x Nk)
        sw: &PyArray2<f64>,       // the actual traces (N x Nk)
        nk: usize,
    ) {
        let x = x.as_array();
        let y = y.as_array();
        let mut sb = unsafe { sb.as_array_mut() };
        let mut sw = unsafe { sw.as_array_mut() };
        lda::get_projection_lda(x, y, &mut sb, &mut sw, nk);
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

    #[pyfn(m, "update_snr_only")]
    fn update_snr_only(
        _py: Python,
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        x: PyReadonlyArray2<u16>,      // (Np,len)
        sum: &PyArray3<i64>,           // (Np,Nc,N_sample)
        sum2: &PyArray3<i64>,          // (Np,Nc,N_sample)
        ns: &PyArray2<u64>,            // (Np,Nc)
    ) {
        let traces = traces.as_array();
        let x = x.as_array();
        let mut sum = unsafe { sum.as_array_mut() };
        let mut sum2 = unsafe { sum2.as_array_mut() };
        let mut ns = unsafe { ns.as_array_mut() };
        snr::update_snr_only(&traces, &x, &mut sum, &mut sum2, &mut ns);
    }

    #[pyfn(m, "finalyze_snr_only")]
    fn finalyze_snr_only(
        _py: Python,
        sum: PyReadonlyArray3<i64>,  // (Np,Nc,N_sample)
        sum2: PyReadonlyArray3<i64>, // (Np,Nc,N_sample)
        ns: PyReadonlyArray2<u64>,   // (Np,Nc,N_sample)
        snr: &PyArray2<f64>,         // (Np,Nc)
    ) {
        let sum = sum.as_array();
        let sum2 = sum2.as_array();
        let ns = ns.as_array();
        let mut snr = unsafe { snr.as_array_mut() };
        snr::finalize_snr_only(&sum, &sum2, &ns, &mut snr);
    }

    Ok(())
}
