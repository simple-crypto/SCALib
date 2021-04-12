//! Belief propagation algorithm implementation.
//!
//! The factor graph has a particular structure: it is made of N copies of an elementary graph,
//! which corresponds to N leaking execution of the same algorithm.
//! Some variable nodes, such as the long-term key, are in the graph only once and are common to
//! all the elementary copies.
//! We call such nodes "single", while the nodes replicated for each copy are "para".
//!
//! The values on the factor graph are probability distribution of values in GF(2)^n.

use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::prelude::{PyResult, Python};
use pyo3::types::PyDict;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::convert::TryInto;

/// Statistical distribution of a Para node.
/// Axes are (id of the copy of the var, value of the field element).
type ParaDistri = Array2<f64>;

/// Statistical distribution of a Single node.
/// Axes are (always length 1, value of the field element).
type SingleDistri = Array2<f64>;

/// Type of a variable node in the factor graph, its initial state and current distribution.
pub enum VarType {
    ProfilePara {
        distri_orig: ParaDistri,
        distri_current: ParaDistri,
    },
    ProfileSingle {
        distri_orig: SingleDistri,
        distri_current: SingleDistri,
    },
    NotProfilePara {
        distri_current: ParaDistri,
    },
    NotProfileSingle {
        distri_current: SingleDistri,
    },
}

/// A variable node.
pub struct Var {
    /// Ids of edges adjacent to the variable node.
    pub neighboors: Vec<usize>,
    pub vartype: VarType,
}

pub enum FuncType {
    /// Bitwise AND of variables
    AND,
    /// Bitwise XOR of variables
    XOR,
    /// Bitwise XOR of variables, XORing additionally a public variable.
    XORCST(Array1<u32>),
    /// Bitwise AND of variables, XORing additionally a public variable.
    ANDCST(Array1<u32>),
    /// Lookup table function.
    LOOKUP(Array1<u32>),
}

/// A function node in the graph.
pub struct Func {
    /// Ids of edges adjacent to the function node.
    pub neighboors: Vec<usize>,
    functype: FuncType,
}

/// The minimum non-zero probability (to avoid denormalization, etc.)
const MIN_PROBA: f64 = 1e-20;

/// Clip down to `MIN_PROBA`
fn make_non_zero<S: ndarray::DataMut + ndarray::RawData<Elem = f64>, D: ndarray::Dimension>(
    x: &mut ndarray::ArrayBase<S, D>,
) {
    x.mapv_inplace(|y| y.max(MIN_PROBA));
}

/// Convert the python description of a variable node to a Var.
pub fn to_var(function: &PyDict) -> Var {
    let neighboors: Vec<isize> = function.get_item("neighboors").unwrap().extract().unwrap();
    let inloop: bool = function.get_item("para").unwrap().extract().unwrap();
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

/// Convert the python description of a function node to a Func.
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
    } else if func == 4 {
        let values: PyReadonlyArray1<u32> = function.get_item("values").unwrap().extract().unwrap();
        f = FuncType::ANDCST(values.as_array().to_owned());
    } else {
        let table: PyReadonlyArray1<u32> = function.get_item("table").unwrap().extract().unwrap();
        f = FuncType::LOOKUP(table.as_array().to_owned());
    }

    Func {
        neighboors: neighboors,
        functype: f,
    }
}

/// Walsh-Hadamard transform (non-normalized).
#[inline(always)]
fn fwht(a: &mut [f64], len: usize) {
    // Note: the speed of this can probably be much improved, with the following techiques
    // * use (auto-)vectorization
    // * generate small static kernels
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

/// Make it such that the sum of the probabilities in the distribution is 1.0.
/// `distri` can be a ParaDistri or a SingleDistri.
fn normalize_distri(distri: &mut Array2<f64>) {
    *distri /= &distri
        .sum_axis(Axis(1))
        .insert_axis(Axis(1))
        .broadcast(distri.shape())
        .unwrap();
}

/// Update `distri` with the information from an `edge`.
fn update_para_var_distri(distri: &mut ParaDistri, edge: &Array2<f64>) {
    *distri *= edge;
    normalize_distri(distri);
}

/// Update the distributions of `variables` based on the messages on `edges` coming from the
/// function nodes.
/// Then, put on `edges` the messages going from the variables to the function nodes.
/// Messages are read from and written to `edges`, where `edges[i][j]` is the message to/from the
/// `j`-th adjacent edge to the variable node `i`.
pub fn update_variables(edges: &mut [Vec<&mut Array2<f64>>], variables: &mut [Var]) {
    variables
        .par_iter_mut()
        .zip(edges.par_iter_mut())
        .for_each(|(var, neighboors)| {
            // update the current distri
            match &mut var.vartype {
                VarType::ProfilePara {
                    distri_orig,
                    distri_current,
                } => {
                    distri_current.assign(&distri_orig);
                    neighboors
                        .iter()
                        .for_each(|msg| update_para_var_distri(distri_current, msg));
                }
                VarType::ProfileSingle {
                    distri_orig,
                    distri_current,
                } => {
                    distri_current.assign(&distri_orig);
                    neighboors.iter().for_each(|msg| {
                        msg.outer_iter().for_each(|msg| {
                            *distri_current *= &msg;
                            normalize_distri(distri_current);
                        });
                    });
                }
                VarType::NotProfilePara { distri_current } => {
                    distri_current.fill(1.0);
                    neighboors
                        .iter()
                        .for_each(|msg| update_para_var_distri(distri_current, msg));
                }
                VarType::NotProfileSingle { distri_current } => {
                    distri_current.fill(1.0);
                    neighboors.iter().for_each(|msg| {
                        msg.outer_iter().for_each(|msg| {
                            *distri_current *= &msg;
                            normalize_distri(distri_current);
                        });
                    });
                }
            }
            // send back the messages
            match &mut var.vartype {
                VarType::ProfilePara { distri_current, .. }
                | VarType::NotProfilePara { distri_current }
                | VarType::ProfileSingle { distri_current, .. }
                | VarType::NotProfileSingle { distri_current } => {
                    neighboors.iter_mut().for_each(|msg| {
                        let distri_current = distri_current.broadcast(msg.shape()).unwrap();
                        msg.zip_mut_with(&distri_current, |msg, distri| *msg = *distri / *msg);
                        normalize_distri(*msg);
                        make_non_zero(msg);
                    });
                    make_non_zero(distri_current);
                }
            }
        });
}

/// Compute the messages from the function nodes to the variable nodes based on the messages from
/// the variable nodes to the function nodes.
/// Messages are read from and written to `edges`, where `edges[i][j]` is the message to/from the
/// `j`-th adjacent edge to the function node `i`.
pub fn update_functions(functions: &[Func], edges: &mut [Vec<&mut Array2<f64>>]) {
    functions
        .par_iter()
        .zip(edges.par_iter_mut())
        .for_each(|(function, edge)| match &function.functype {
            FuncType::AND => {
                let [output_msg, input1_msg, input2_msg]: &mut [_; 3] =
                    edge.as_mut_slice().try_into().unwrap();
                let nc = input1_msg.shape()[1];
                (
                    input1_msg.outer_iter_mut(),
                    input2_msg.outer_iter_mut(),
                    output_msg.outer_iter_mut(),
                )
                    .into_par_iter()
                    // Use for_each_init to limit the number of a allocation of the message
                    // scratch-pad.
                    .for_each_init(
                        || (Array1::zeros(nc), Array1::zeros(nc), Array1::zeros(nc)),
                        |(in1_msg_scratch, in2_msg_scratch, out_msg_scratch),
                         (mut input1_msg, mut input2_msg, mut output_msg)| {
                            in1_msg_scratch.fill(0.0);
                            in2_msg_scratch.fill(0.0);
                            out_msg_scratch.fill(0.0);

                            for i1 in 0..nc {
                                for i2 in 0..nc {
                                    let o: usize = i1 & i2;
                                    in1_msg_scratch[i1] += input2_msg[i2] * output_msg[o];
                                    in2_msg_scratch[i2] += input1_msg[i1] * output_msg[o];
                                    out_msg_scratch[o] += input1_msg[i1] * input2_msg[i2];
                                }
                            }
                            input1_msg.assign(in1_msg_scratch);
                            input2_msg.assign(in2_msg_scratch);
                            output_msg.assign(out_msg_scratch);
                        },
                    );
            }
            FuncType::XOR => {
                xors(edge.as_mut());
            }
            FuncType::XORCST(values) => {
                let [output_msg, input1_msg]: &mut [_; 2] = edge.as_mut_slice().try_into().unwrap();
                let nc = input1_msg.shape()[1];
                (
                    input1_msg.outer_iter_mut(),
                    output_msg.outer_iter_mut(),
                    values.outer_iter(),
                )
                    .into_par_iter()
                    .for_each_init(
                        || (Array1::zeros(nc), Array1::zeros(nc)),
                        |(in1_msg_scratch, out_msg_scratch),
                         (mut input1_msg, mut output_msg, value)| {
                            in1_msg_scratch.fill(0.0);
                            out_msg_scratch.fill(0.0);
                            let value = value.first().unwrap();
                            for i1 in 0..nc {
                                let o: usize = ((i1 as u32) ^ value) as usize;
                                in1_msg_scratch[i1] += output_msg[o];
                                out_msg_scratch[o] += input1_msg[i1];
                            }
                            input1_msg.assign(in1_msg_scratch);
                            output_msg.assign(out_msg_scratch);
                        },
                    );
            }
            FuncType::ANDCST(values) => {
                let [output_msg, input1_msg]: &mut [_; 2] = edge.as_mut_slice().try_into().unwrap();
                let nc = input1_msg.shape()[1];
                (
                    input1_msg.outer_iter_mut(),
                    output_msg.outer_iter_mut(),
                    values.outer_iter(),
                )
                    .into_par_iter()
                    .for_each_init(
                        || (Array1::zeros(nc), Array1::zeros(nc)),
                        |(in1_msg_scratch, out_msg_scratch),
                         (mut input1_msg, mut output_msg, value)| {
                            in1_msg_scratch.fill(0.0);
                            out_msg_scratch.fill(0.0);
                            let value = value.first().unwrap();
                            for i1 in 0..nc {
                                let o: usize = ((i1 as u32) & value) as usize;
                                in1_msg_scratch[i1] += output_msg[o];
                                out_msg_scratch[o] += input1_msg[i1];
                            }
                            input1_msg.assign(in1_msg_scratch);
                            output_msg.assign(out_msg_scratch);
                        },
                    );
            }
            FuncType::LOOKUP(table) => {
                let [output_msg, input1_msg]: &mut [_; 2] = edge.as_mut_slice().try_into().unwrap();
                let nc = input1_msg.shape()[1];
                (input1_msg.outer_iter_mut(), output_msg.outer_iter_mut())
                    .into_par_iter()
                    .for_each_init(
                        || (Array1::zeros(nc), Array1::zeros(nc)),
                        |(in1_msg_scratch, out_msg_scratch), (mut input1_msg, mut output_msg)| {
                            in1_msg_scratch.fill(0.0);
                            out_msg_scratch.fill(0.0);
                            for i1 in 0..nc {
                                let o: usize = table[i1] as usize;
                                // This requires table to be bijective. Otherwise, we would have to
                                // divide the messge on the output by the number of matching inputs
                                // to get the message to forward on the input edge.
                                in1_msg_scratch[i1] += output_msg[o];
                                out_msg_scratch[o] += input1_msg[i1];
                            }
                            input1_msg.assign(in1_msg_scratch);
                            output_msg.assign(out_msg_scratch);
                        },
                    );
            }
        });
}

/// Compute a XOR function node between all edges.
pub fn xors(inputs: &mut [&mut Array2<f64>]) {
    let n_runs = inputs[0].shape()[0];
    let nc = inputs[0].shape()[1];
    for run in 0..n_runs {
        let mut acc = Array1::<f64>::ones(nc);
        // Accumulate in a Walsh transformed domain.
        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![run, ..]);
            let input_fwt_s = input.as_slice_mut().unwrap();
            fwht(input_fwt_s, nc);
            // non zero with input_fwt_s possibly negative
            input.mapv_inplace(|x| {
                if x.is_sign_positive() {
                    x.max(MIN_PROBA)
                } else {
                    x.min(-MIN_PROBA)
                }
            });
            acc.zip_mut_with(&input, |x, y| *x = *x * y);
            acc /= acc.sum();
        });
        // Invert accumulation input-wise and invert transform.
        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![run, ..]);
            input.zip_mut_with(&acc, |x, y| *x = *y / *x);
            let input_fwt_s = input.as_slice_mut().unwrap();
            fwht(input_fwt_s, nc);
            make_non_zero(&mut input);
            let s = input.sum();
            input /= s;
            make_non_zero(&mut input);
        });
    }
}

/// Run the belief propagation algorithm on the python representation of a factor graph.
#[pyfunction]
pub fn run_bp(
    py: Python,
    functions: &PyList,
    variables: &PyList,
    it: usize,
    // number of variable nodes in the graph
    edge: usize,
    // size of the field
    nc: usize,
    // number of copies in the graph (n_runs)
    n: usize,
) -> PyResult<()> {
    // Scratch array containing all the edge's messages.
    let mut edges: Vec<Array2<f64>> = vec![Array2::<f64>::ones((n, nc)); edge];

    // Mapping of each edge to its (function node id, position in function node).
    let mut vec_funcs_id: Vec<(usize, usize)> = vec![(0, 0); edge];
    // Mapping of each edge to its (variable node id, position in variable node).
    let mut vec_vars_id: Vec<(usize, usize)> = vec![(0, 0); edge];

    // map all python functions to rust ones + generate the mapping in vec_functs_id
    let functions_rust: Vec<Func> = functions
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let dict = x.downcast::<PyDict>().unwrap();
            let f = to_func(dict);
            f.neighboors.iter().enumerate().for_each(|(j, x)| {
                vec_funcs_id[*x] = (i, j);
            });
            f
        })
        .collect();

    // map all python var to rust ones
    // generate the edge mapping in vec_vars_id
    // init the messages along the edges with initial distributions
    let mut variables_rust: Vec<Var> = variables
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let dict = x.downcast::<PyDict>().unwrap();
            let var = to_var(dict);
            var.neighboors.iter().enumerate().for_each(|(j, x)| {
                vec_vars_id[*x] = (i, j);
            });
            match &var.vartype {
                VarType::ProfilePara { distri_orig, .. }
                | VarType::ProfileSingle { distri_orig, .. } => {
                    var.neighboors.iter().for_each(|x| {
                        let v = &mut edges[*x];
                        let distri_orig = distri_orig.broadcast(v.shape()).unwrap();
                        v.assign(&distri_orig);
                    })
                }
                _ => {}
            }
            var
        })
        .collect();

    py.allow_threads(|| {
        // loading bar
        let pb = ProgressBar::new(it as u64);
        pb.set_style(ProgressStyle::default_spinner().template(
        "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})"));
        pb.set_message("Calculating BP...");

        for _ in (0..it).progress_with(pb) {
            // This is a technique for runtime borrow-checking: we take reference on all the edges
            // at once, put them into options, then extract the references out of the options, one
            // at a time and out-of-order.
            let mut edge_opt_ref_mut: Vec<Option<&mut Array2<f64>>> =
                edges.iter_mut().map(|x| Some(x)).collect();
            let mut edge_for_func: Vec<Vec<&mut Array2<f64>>> = functions_rust
                .iter()
                .map(|f| {
                    f.neighboors
                        .iter()
                        .map(|e| edge_opt_ref_mut[*e].take().unwrap())
                        .collect()
                })
                .collect();
            update_functions(&functions_rust, &mut edge_for_func);
            let mut edge_opt_ref_mut: Vec<Option<&mut Array2<f64>>> =
                edges.iter_mut().map(|x| Some(x)).collect();
            let mut edge_for_var: Vec<Vec<&mut Array2<f64>>> = variables_rust
                .iter()
                .map(|f| {
                    f.neighboors
                        .iter()
                        .map(|e| edge_opt_ref_mut[*e].take().unwrap())
                        .collect()
                })
                .collect();
            update_variables(&mut edge_for_var, &mut variables_rust);
        }
    });

    variables_rust
        .iter()
        .zip(variables)
        .for_each(|(v_rust, v_python)| {
            let distri_current = match &v_rust.vartype {
                VarType::NotProfilePara {
                    distri_current: distri,
                }
                | VarType::NotProfileSingle {
                    distri_current: distri,
                }
                | VarType::ProfilePara {
                    distri_current: distri,
                    ..
                }
                | VarType::ProfileSingle {
                    distri_current: distri,
                    ..
                } => distri,
            };
            v_python
                .set_item("current", PyArray2::from_array(py, &distri_current))
                .unwrap();
        });
    Ok(())
}
