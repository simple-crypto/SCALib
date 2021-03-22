use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use stella::belief_propagation;
use stella::lda;
use stella::snr;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::{PyDict, PyList};

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn str2method(s: &str) -> Result<ranklib::RankingMethod, &str> {
    match s {
        "naive" => Ok(ranklib::RankingMethod::Naive),
        "hist" => Ok(ranklib::RankingMethod::Hist),
        #[cfg(feature = "ntl")]
        "histbignum" => Ok(ranklib::RankingMethod::HistBigNum),
        #[cfg(feature = "hellib")]
        "hellib" => Ok(ranklib::RankingMethod::Hellib),
        _ => Err("Invalid method name"),
    }
}

#[pymodule]
fn stella(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<snr::SNR>()?;
    m.add_class::<lda::LDA>()?;
    #[pyfn(m, "belief_propagation")]
    fn belief_propagation(
        py: Python,
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
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
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
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
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

        py.allow_threads(|| {
            // loading bar
            let pb = ProgressBar::new(it as u64);
            pb.set_style(ProgressStyle::default_spinner().template(
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"
    ));
            pb.set_message("Calculating BP...");

            for _ in (0..it).progress_with(pb) {
                unsafe {
                    // map vertex to vec<vec<>> based on vec_funcs_id
                    let mut vertex_for_func: Vec<Vec<&mut Array2<f64>>> = functions_rust
                        .iter()
                        .map(|v| {
                            let mut vec =
                                Vec::<&mut Array2<f64>>::with_capacity(v.neighboors.len());
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
                            let mut vec =
                                Vec::<&mut Array2<f64>>::with_capacity(v.neighboors.len());
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
        });

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
    #[pyfn(m, "partial_cp")]
    fn partial_cp(
        _py: Python,
        traces: PyReadonlyArray2<i16>, // (len,N_sample)
        poi: PyReadonlyArray1<u32>,    // (Np,len)
        store: &PyArray2<i16>,
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

    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    #[pyfn(m, "rank_accuracy")]
    fn rank_accuracy(
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        acc: f64,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        let res = str2method(&method);
        match res {
            Ok(res) => {
                let res = res.rank_accuracy(&costs, &key, acc, merge);
                match res {
                    Ok(res) => Ok((res.min, res.est, res.max)),
                    Err(s) => {
                        println!("{}", s);
                        panic!()
                    }
                }
            }
            Err(_) => panic!(),
        }
        //return Ok((res.min, res.est, res.max));
    }

    #[pyfn(m, "rank_nbin")]
    fn rank_nbin(
        costs: Vec<Vec<f64>>,
        key: Vec<usize>,
        nb_bin: usize,
        merge: Option<usize>,
        method: String,
    ) -> PyResult<(f64, f64, f64)> {
        let res = str2method(&method);
        match res {
            Ok(res) => {
                let res = res.rank_nbin(&costs, &key, nb_bin, merge);
                match res {
                    Ok(res) => Ok((res.min, res.est, res.max)),
                    Err(s) => {
                        println!("{}", s);
                        panic!()
                    }
                }
            }
            Err(_) => panic!(),
        }
    }


    Ok(())
}
