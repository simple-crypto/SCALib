extern crate ndarray;
extern crate openblas_src;
mod belief_propagation;
mod lda;
mod snr;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Axis};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
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
        c_means_bak: &PyArray2<f64>, // the actual traces (N x Nk)
        x_f64: &PyArray2<f64>,
        nk: usize,
    ) {
        let x = x.as_array();
        let y = y.as_array();
        let mut sb = unsafe { sb.as_array_mut() };
        let mut sw = unsafe { sw.as_array_mut() };
        let mut c_means_bak = unsafe { c_means_bak.as_array_mut() };
        let mut x_f64 = unsafe { x_f64.as_array_mut() };
        lda::get_projection_lda(x, y, &mut sb, &mut sw, &mut c_means_bak, &mut x_f64, nk);
    }

    #[pyfn(m, "predict_proba_lda")]
    fn predict_proba_lda(
        _py: Python,
        x: PyReadonlyArray2<i16>, // U matrix (decomposition of Inv Cov (Npro x Npro)
        projection: PyReadonlyArray2<f64>,
        c_means: PyReadonlyArray2<f64>,
        psd: PyReadonlyArray2<f64>,
        prs: &PyArray2<f64>,
    ) {
        let x = x.as_array();
        let projection = projection.as_array();
        let c_means = c_means.as_array();
        let psd = psd.as_array();
        let mut prs = unsafe { prs.as_array_mut() };
        lda::predict_proba_lda(x, projection, c_means, psd, &mut prs);
    }

    #[pyfn(m, "partial_cp")]
    fn partial_cp<T>(
        _py: Python,
        traces: PyReadonlyArray2<T>, // (len,N_sample)
        poi: PyReadonlyArray1<u32>,  // (Np,len)
        store: &PyArray2<T>,
    ) {
        let traces = traces.as_array();
        let poi = poi.as_array();
        let mut store = unsafe { store.as_array_mut() };
        store
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(poi.outer_iter().into_par_iter())
            .for_each(|(mut x, poi)| {
                let poi = poi.first().unwrap();
                x.assign(&traces.slice(s![.., *poi as usize]));
            });
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
