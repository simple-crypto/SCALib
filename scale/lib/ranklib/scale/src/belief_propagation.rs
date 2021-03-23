use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::prelude::{PyResult, Python};
use pyo3::types::PyDict;
use pyo3::types::PyList;
use rayon::prelude::*;

pub enum VarType {
    ProfilePara {
        distri_orig: Array2<f64>,
        distri_current: Array2<f64>,
    },
    ProfileSingle {
        distri_orig: Array2<f64>,
        distri_current: Array2<f64>,
    },
    NotProfilePara {
        distri_current: Array2<f64>,
    },
    NotProfileSingle {
        distri_current: Array2<f64>,
    },
}

pub struct Var {
    pub neighboors: Vec<usize>, // (id,offset)
    pub vartype: VarType,
}

pub enum FuncType {
    AND,
    XOR,
    XORCST(Array1<u32>),
    LOOKUP(Array1<u32>),
}
pub struct Func {
    pub neighboors: Vec<usize>,
    functype: FuncType,
}

pub fn to_var(function: &PyDict) -> Var {
    let neighboors: Vec<isize> = function.get_item("neighboors").unwrap().extract().unwrap();
    let inloop: bool = function.get_item("in_loop").unwrap().extract().unwrap();
    let is_profiled = function.contains("initial").unwrap();
    let distri_current: PyReadonlyArray2<f64> =
        function.get_item("current").unwrap().extract().unwrap();

    let neighboors: Vec<usize> = neighboors.iter().map(|x| *x as usize).collect();
    let f: VarType;
    if inloop & is_profiled {
        let distri_orig: PyReadonlyArray2<f64> =
            function.get_item("initial").unwrap().extract().unwrap();
        f = VarType::ProfilePara {
            distri_orig: distri_orig.as_array().to_owned(),
            distri_current: distri_orig.as_array().to_owned(),
        };
    } else if inloop & !is_profiled {
        f = VarType::NotProfilePara {
            distri_current: distri_current.as_array().to_owned(),
        };
    } else if !inloop & is_profiled {
        let distri_orig: PyReadonlyArray2<f64> =
            function.get_item("initial").unwrap().extract().unwrap();
        f = VarType::ProfileSingle {
            distri_orig: distri_orig.as_array().to_owned(),
            distri_current: distri_orig.as_array().to_owned(),
        };
    } else {
        f = VarType::NotProfileSingle {
            distri_current: distri_current.as_array().to_owned(),
        };
    }

    Var {
        neighboors: neighboors,
        vartype: f,
    }
}

pub fn to_func(function: &PyDict) -> Func {
    let neighboors: Vec<isize> = function.get_item("neighboors").unwrap().extract().unwrap();
    let func: usize = function.get_item("func").unwrap().extract().unwrap();

    let neighboors: Vec<usize> = neighboors.iter().map(|x| *x as usize).collect();

    let f: FuncType;
    if func == 0 {
        f = FuncType::AND;
    } else if func == 1 {
        f = FuncType::XOR;
    } else if func == 2 {
        let values: PyReadonlyArray1<u32> = function.get_item("values").unwrap().extract().unwrap();
        f = FuncType::XORCST(values.as_array().to_owned());
    } else {
        let table: PyReadonlyArray1<u32> = function.get_item("table").unwrap().extract().unwrap();
        f = FuncType::LOOKUP(table.as_array().to_owned());
    }

    Func {
        neighboors: neighboors,
        functype: f,
    }
}

#[inline(always)]
fn fwht(a: &mut [f64], len: usize) {
    let mut h = 1;
    while h < len {
        for mut i in 0..(len / (2 * h) as usize) {
            i *= 2 * h;
            for j in i..(i + h) {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

pub fn update_variables(vertex: &mut Vec<Vec<&mut Array2<f64>>>, variables: &mut Vec<Var>) {
    variables
        .par_iter_mut()
        .zip(vertex.par_iter_mut())
        .for_each(|(var, neighboors)| {
            // update the current distri
            match &mut var.vartype {
                VarType::ProfilePara {
                    distri_orig,
                    distri_current,
                } => {
                    distri_current.assign(&distri_orig);
                    neighboors.iter().for_each(|msg| {
                        distri_current.zip_mut_with(msg, |x, y| *x *= *y);
                        *distri_current /= &distri_current
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(distri_current.shape())
                            .unwrap();
                    });
                }
                VarType::ProfileSingle {
                    distri_orig,
                    distri_current,
                } => {
                    distri_current.assign(&distri_orig);
                    neighboors.iter().for_each(|msg| {
                        msg.outer_iter().for_each(|msg| {
                            *distri_current *= &msg;
                            *distri_current /= distri_current.fold(0.0, |acc, x| acc + *x);
                        });
                    });
                }
                VarType::NotProfilePara { distri_current } => {
                    distri_current.fill(1.0);
                    neighboors.iter().for_each(|msg| {
                        distri_current.zip_mut_with(msg, |x, y| *x *= *y);
                        *distri_current /= &distri_current
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(distri_current.shape())
                            .unwrap();
                    });
                }
                VarType::NotProfileSingle { distri_current } => {
                    distri_current.fill(1.0);
                    neighboors.iter().for_each(|msg| {
                        msg.outer_iter().for_each(|msg| {
                            *distri_current *= &msg;
                            *distri_current /= distri_current.fold(0.0, |acc, x| acc + *x);
                        });
                    });
                }
            }

            // send back the messages
            match &mut var.vartype {
                VarType::ProfilePara {
                    distri_orig: _,
                    distri_current,
                } => {
                    neighboors.iter_mut().for_each(|msg| {
                        msg.zip_mut_with(distri_current, |msg, distri| *msg = *distri / *msg);
                        **msg /= &msg
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(msg.shape())
                            .unwrap();
                        msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                    });
                }
                VarType::ProfileSingle {
                    distri_orig: _,
                    distri_current,
                } => {
                    neighboors.iter_mut().for_each(|msg| {
                        let distri_current = distri_current.broadcast(msg.shape()).unwrap();
                        msg.zip_mut_with(&distri_current, |msg, distri| *msg = *distri / *msg);
                        **msg /= &msg
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(msg.shape())
                            .unwrap();
                        msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                    });
                }
                VarType::NotProfilePara { distri_current } => {
                    neighboors.iter_mut().for_each(|msg| {
                        msg.zip_mut_with(distri_current, |msg, distri| *msg = *distri / *msg);
                        **msg /= &msg
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(msg.shape())
                            .unwrap();
                        msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                    });
                }
                VarType::NotProfileSingle { distri_current } => {
                    neighboors.iter_mut().for_each(|msg| {
                        let distri_current = distri_current.broadcast(msg.shape()).unwrap();
                        msg.zip_mut_with(&distri_current, |msg, distri| *msg = *distri / *msg);
                        **msg /= &msg
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(msg.shape())
                            .unwrap();
                        msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                    });
                }
            }
        });
}

pub fn update_functions(functions: &mut Vec<Func>, vertex: &mut Vec<Vec<&mut Array2<f64>>>) {
    functions
        .par_iter_mut()
        .zip(vertex.par_iter_mut())
        .for_each(|(function, mut vertex)| {
            match &mut function.functype {
                FuncType::AND => {
                    let input2_msg = vertex.pop().unwrap();
                    let input1_msg = vertex.pop().unwrap();
                    let output_msg = vertex.pop().unwrap();
                    let nc = input1_msg.shape()[1];
                    input1_msg
                        .outer_iter_mut()
                        .into_par_iter()
                        .zip(input2_msg.outer_iter_mut().into_par_iter())
                        .zip(output_msg.outer_iter_mut().into_par_iter())
                        .for_each(|((mut input1_msg, mut input2_msg), mut output_msg)| {
                            let input1_msg_o = input1_msg.to_owned();
                            let input2_msg_o = input2_msg.to_owned();
                            let output_msg_o = output_msg.to_owned();
                            let input1_msg_s = input1_msg_o.as_slice().unwrap();
                            let input2_msg_s = input2_msg_o.as_slice().unwrap();
                            let output_msg_s = output_msg_o.as_slice().unwrap();

                            input1_msg.fill(0.0);
                            input2_msg.fill(0.0);
                            output_msg.fill(0.0);
                            let input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                            let input2_msg_s_mut = input2_msg.as_slice_mut().unwrap();
                            let output_msg_s_mut = output_msg.as_slice_mut().unwrap();

                            for i1 in 0..nc {
                                for i2 in 0..nc {
                                    let o: usize = i1 & i2;
                                    // input 1
                                    input1_msg_s_mut[i1] += input2_msg_s[i2] * output_msg_s[o];
                                    // input 2
                                    input2_msg_s_mut[i2] += input1_msg_s[i1] * output_msg_s[o];
                                    // out
                                    output_msg_s_mut[o] += input1_msg_s[i1] * input2_msg_s[i2];
                                }
                            }
                        });
                }
                FuncType::XOR => {
                    let nc = vertex[0].shape()[1];
                    xors(&mut vertex, nc);
                }
                FuncType::XORCST(values) => {
                    let input1_msg = vertex.pop().unwrap();
                    let output_msg = vertex.pop().unwrap();
                    let nc = input1_msg.shape()[1];
                    input1_msg
                        .outer_iter_mut()
                        .into_par_iter()
                        .zip(output_msg.outer_iter_mut().into_par_iter())
                        .zip(values.outer_iter().into_par_iter())
                        .for_each(|((mut input1_msg, mut output_msg), value)| {
                            let input1_msg_o = input1_msg.to_owned();
                            let output_msg_o = output_msg.to_owned();
                            let input1_msg_s = input1_msg_o.as_slice().unwrap();
                            let output_msg_s = output_msg_o.as_slice().unwrap();

                            input1_msg.fill(0.0);
                            output_msg.fill(0.0);
                            let input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                            let output_msg_s_mut = output_msg.as_slice_mut().unwrap();
                            let value = value.first().unwrap();
                            for i1 in 0..nc {
                                let o: usize = ((i1 as u32) ^ value) as usize;
                                input1_msg_s_mut[i1] += output_msg_s[o];
                                output_msg_s_mut[o] += input1_msg_s[i1];
                            }
                        });
                }
                FuncType::LOOKUP(table) => {
                    let input1_msg = vertex.pop().unwrap();
                    let output_msg = vertex.pop().unwrap();
                    let nc = input1_msg.shape()[1];
                    let table = table.as_slice().unwrap();
                    input1_msg
                        .outer_iter_mut()
                        .into_par_iter()
                        .zip(output_msg.outer_iter_mut().into_par_iter())
                        .for_each(|(mut input1_msg, mut output_msg)| {
                            let input1_msg_o = input1_msg.to_owned();
                            let output_msg_o = output_msg.to_owned();
                            let input1_msg_s = input1_msg_o.as_slice().unwrap();
                            let output_msg_s = output_msg_o.as_slice().unwrap();

                            input1_msg.fill(0.0);
                            output_msg.fill(0.0);
                            let input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                            let output_msg_s_mut = output_msg.as_slice_mut().unwrap();

                            for i1 in 0..nc {
                                let o: usize = table[i1] as usize;
                                input1_msg_s_mut[i1] += output_msg_s[o];
                                output_msg_s_mut[o] += input1_msg_s[i1];
                            }
                        });
                }
            }
        });
}

pub fn xors(inputs: &mut Vec<&mut Array2<f64>>, nc: usize) {
    for i in 0..inputs[0].shape()[0] {
        let mut acc = Array1::<f64>::ones(nc);

        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![i, ..]);
            let input_fwt_s = input.as_slice_mut().unwrap();
            fwht(input_fwt_s, nc);
            input_fwt_s
                .iter_mut()
                .for_each(|x| *x = if f64::abs(*x) == 0.0 { 1E-50 } else { *x });
            acc.zip_mut_with(&input, |x, y| *x = *x * y);
            acc /= acc.sum();
        });

        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![i, ..]);
            input.zip_mut_with(&acc, |x, y| *x = *y / *x);
            let input_fwt_s = input.as_slice_mut().unwrap();
            fwht(input_fwt_s, nc);
            let s = input.iter().fold(0.0, |acc, x| acc + x.max(1E-50));
            input
                .iter_mut()
                .for_each(|x| *x = (x.max(1E-50) / s).max(1E-50));
        });
    }
}

#[pyfunction]
pub fn run_bp(
    py: Python,
    functions: &PyList,
    variables: &PyList,
    it: usize,
    vertex: usize,
    nc: usize,
    n: usize,
) -> PyResult<()> {
    // array to save all the vertex
    let mut vertex: Vec<Array2<f64>> = (0..vertex).map(|_| Array2::<f64>::ones((n, nc))).collect();

    // mapping of the vertex for functions and variables
    let mut vec_funcs_id: Vec<(usize, usize)> = (0..vertex.len()).map(|_| (0, 0)).collect(); //(associated funct,position in fnc)
    let mut vec_vars_id: Vec<(usize, usize)> = (0..vertex.len()).map(|_| (0, 0)).collect();

    // loading bar
    let pb = ProgressBar::new(functions.len() as u64);
    pb.set_style(ProgressStyle::default_spinner().template(
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
    ));
    pb.set_message("Init functions...");

    // map all python functions to rust ones + generate the mapping in vec_functs_id
    let mut functions_rust: Vec<Func> = functions
        .iter()
        .enumerate()
        .progress_with(pb)
        .map(|(i, x)| {
            let dict = x.downcast::<PyDict>().unwrap();
            let f = to_func(dict);
            f.neighboors.iter().enumerate().for_each(|(j, x)| {
                vec_funcs_id[*x] = (i, j);
            });
            f
        })
        .collect();

    // loading bar
    let pb = ProgressBar::new(variables.len() as u64);
    pb.set_style(ProgressStyle::default_spinner().template(
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
    ));
    pb.set_message("Init variables...");

    // map all python var to rust ones
    // generate the vertex mapping in vec_vars_id
    // init the messages along the edges with initial distributions
    let mut variables_rust: Vec<Var> = variables
        .iter()
        .progress_with(pb)
        .enumerate()
        .map(|(i, x)| {
            let dict = x.downcast::<PyDict>().unwrap();
            let var = to_var(dict);
            match &var.vartype {
                VarType::ProfilePara {
                    distri_orig,
                    distri_current: _,
                } => var.neighboors.iter().enumerate().for_each(|(j, x)| {
                    let v = &mut vertex[*x];
                    v.assign(&distri_orig);
                    vec_vars_id[*x] = (i, j);
                }),

                VarType::ProfileSingle {
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
                update_functions(&mut functions_rust, &mut vertex_for_func);
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
                update_variables(&mut vertex_for_var, &mut variables_rust);
            }
        }
    });

    let pb = ProgressBar::new(variables.len() as u64);
    pb.set_style(ProgressStyle::default_spinner().template(
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
    ));
    pb.set_message("Dump variables...");
    variables_rust
        .iter()
        .progress_with(pb)
        .zip(variables)
        .for_each(|(v_rust, v_python)| {
            let distri_current: &PyArray2<f64> =
                v_python.get_item("current").unwrap().extract().unwrap();
            let mut distri_current = unsafe { distri_current.as_array_mut() };
            match &v_rust.vartype {
                VarType::NotProfilePara {
                    distri_current: distri,
                } => {
                    distri_current.assign(&distri);
                }
                VarType::NotProfileSingle {
                    distri_current: distri,
                } => {
                    distri_current.assign(&distri);
                }
                VarType::ProfilePara {
                    distri_orig: _,
                    distri_current: distri,
                } => {
                    distri_current.assign(&distri);
                }
                VarType::ProfileSingle {
                    distri_orig: _,
                    distri_current: distri,
                } => {
                    distri_current.assign(&distri);
                }
            }
        });
    Ok(())
}
