//! Belief propagation algorithm implementation.
//!
//! The factor graph has a particular structure: it is made of N copies of an elementary graph,
//! which corresponds to N leaking execution of the same algorithm.
//! Some variable nodes, such as the long-term key, are in the graph only once and are common to
//! all the elementary copies.
//! We call such nodes "single", while the nodes replicated for each copy are "para".
//!
//! The values on the factor graph are probability distribution of values in GF(2)^n.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use rayon::prelude::*;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
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
    /// Bitwise NOT of variables
    NOT,
    /// Modular ADD of variables
    ADD,
    /// Modular MUL of variables
    MUL,
    /// Bitwise XOR of variables, XORing additionally a public variable.
    XORCST(Array1<u32>),
    /// Bitwise AND of variables, ANDing additionally a public variable.
    ANDCST(Array1<u32>),
    /// Modular ADD of variables, ADDing additionally a public variable.
    ADDCST(Array1<u32>),
    /// Modular MUL of variables, MULing additionally a public variable.
    MULCST(Array1<u32>),
    /// Lookup table function.
    LOOKUP { table: Array1<u32> },
}

/// A function node in the graph.
pub struct Func {
    /// Ids of edges adjacent to the function node.
    pub neighboors: Vec<usize>,
    pub functype: FuncType,
}

/// The minimum non-zero probability (to avoid denormalization, etc.)
const MIN_PROBA: f64 = 1e-40;

/// Clip down to `MIN_PROBA`
fn make_non_zero<S: ndarray::DataMut + ndarray::RawData<Elem = f64>, D: ndarray::Dimension>(
    x: &mut ndarray::ArrayBase<S, D>,
) {
    x.mapv_inplace(|y| y.max(MIN_PROBA));
}

fn make_non_zero_signed<
    S: ndarray::DataMut + ndarray::RawData<Elem = f64>,
    D: ndarray::Dimension,
>(
    x: &mut ndarray::ArrayBase<S, D>,
) {
    x.mapv_inplace(|y| {
        if y.is_sign_positive() {
            y.max(MIN_PROBA)
        } else {
            y.min(-MIN_PROBA)
        }
    });
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

/// Cumulative transform (U in RLDA paper)
#[inline(always)]
fn cumt(a: &mut [f64], len: usize) {
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
                a[j + h] = y;
            }
        }
        h *= 2;
    }
}

/// Cumulative inverse transform (U^-1 in RLDA paper)
#[inline(always)]
fn cumti(a: &mut [f64], len: usize) {
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
                a[j] = x - y;
                a[j + h] = y;
            }
        }
        h *= 2;
    }
}

/// Tansform for operand of AND (V in RLDA paper), involutive
#[inline(always)]
fn opandt(a: &mut [f64], len: usize) {
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
                a[j] = x;
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
            // TODO: if nc is prime, the update for MUL can be computed more efficiently by mapping
            // classes to their discrete logarithm, and by applying FFT.
            // If not prime, use CRT.
            FuncType::MUL => {
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
                                    // Unifies operators that can only be binary
                                    let o = (((i1 * i2) as u32) % (nc as u32)) as usize;
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
            FuncType::AND => {
                ands(edge.as_mut());
            }
            FuncType::ADD => {
                adds(edge.as_mut());
            }
            FuncType::XOR => {
                xors(edge.as_mut());
            }
            FuncType::NOT => {
                let [output_msg, input_msg]: &mut [_; 2] = edge.as_mut_slice().try_into().unwrap();
                let nc = input_msg.shape()[1];
                (input_msg.outer_iter_mut(), output_msg.outer_iter_mut())
                    .into_par_iter()
                    .for_each_init(
                        || (Array1::zeros(nc), Array1::zeros(nc)),
                        |(in_msg_scratch, out_msg_scratch), (mut input_msg, mut output_msg)| {
                            for i in 0..nc {
                                let o = (nc - 1) ^ i as usize;
                                in_msg_scratch[i] = output_msg[o];
                                out_msg_scratch[o] = input_msg[i];
                            }
                            input_msg.assign(in_msg_scratch);
                            output_msg.assign(out_msg_scratch);
                        },
                    );
            }
            FuncType::XORCST(values)
            | FuncType::ANDCST(values)
            | FuncType::ADDCST(values)
            | FuncType::MULCST(values) => {
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
                                let o: usize = match &function.functype {
                                    FuncType::XORCST(_) => ((i1 as u32) ^ value) as usize,
                                    FuncType::ANDCST(_) => ((i1 as u32) & value) as usize,
                                    FuncType::ADDCST(_) => {
                                        (((i1 as u32) + value) % (nc as u32)) as usize
                                    }
                                    FuncType::MULCST(_) => {
                                        (((i1 as u32) * value) % (nc as u32)) as usize
                                    }
                                    _ => unreachable!(),
                                };
                                in1_msg_scratch[i1] += output_msg[o];
                                out_msg_scratch[o] += input1_msg[i1];
                            }
                            input1_msg.assign(in1_msg_scratch);
                            output_msg.assign(out_msg_scratch);
                        },
                    );
            }
            FuncType::LOOKUP { table } => {
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

/// Compute an ADD function node between all edges.
pub fn adds(inputs: &mut [&mut Array2<f64>]) {
    let n_runs = inputs[0].shape()[0];
    let nc = inputs[0].shape()[1];

    // Sets the FFT operator
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(nc);
    let c2r = real_planner.plan_fft_inverse(nc);

    for run in 0..n_runs {
        let mut spectrums: Vec<Array1<Complex<f64>>> = Vec::new();
        let mut acc = Array1::<Complex<f64>>::ones(nc / 2 + 1);
        inputs.iter_mut().for_each(|input| {
            let mut input = input.slice_mut(s![run, ..]);
            let input_fft_s = input.as_slice_mut().unwrap();
            let mut spectrum = Array1::<Complex<f64>>::zeros(nc / 2 + 1);
            let spec = spectrum.as_slice_mut().unwrap();
            // Computes the FFT
            r2c.process(input_fft_s, spec).unwrap();
            // Clips the transformed
            spectrum.mapv_inplace(|x| {
                if x.norm_sqr() == 0.0 {
                    Complex::new(MIN_PROBA, MIN_PROBA)
                } else {
                    x
                }
            });
            spectrums.push(spectrum);
            // Accumulates through the operands
            acc.zip_mut_with(&spectrums[spectrums.len() - 1], |x, y| *x = *x * y);
            acc /= acc.sum();
        });
        assert_eq!(inputs.len(), spectrums.len());
        // Invert accumulation input_wise and invert transform.
        spectrums
            .iter_mut()
            .zip(inputs.iter_mut())
            .for_each(|(spectrum, input)| {
                let mut input = input.slice_mut(s![run, ..]);
                spectrum.zip_mut_with(&acc, |x, y| *x = *y / *x);
                let input_fft_s = input.as_slice_mut().unwrap();
                let spec = spectrum.as_slice_mut().unwrap();
                c2r.process(spec, input_fft_s).unwrap();
                make_non_zero(&mut input);
                let s = input.sum();
                input /= s;
                make_non_zero(&mut input);
            });
    }
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

/// Compute a AND function node between all edges.
pub fn ands(inputs: &mut [&mut Array2<f64>]) {
    let n_runs = inputs[0].shape()[0];
    let nc = inputs[0].shape()[1];
    let (input_0, input_1, input_2) = match inputs {
        [i0, i1, i2] => (i0, i1, i2),
        _ => panic!("Current implementation limit AND to 2 inputs."),
    };
    for run in 0..n_runs {
        let mut cum_t1 = input_1.slice(s![run, ..]).to_owned();
        let mut cum_t2 = input_2.slice(s![run, ..]).to_owned();
        let cum_transform = |mut x: ArrayViewMut1<f64>| {
            let x_slice = x.as_slice_mut().unwrap();
            cumt(x_slice, nc);
            make_non_zero(&mut x);
        };
        let opand_transform = |mut x: ArrayViewMut1<f64>| {
            let x_slice = x.as_slice_mut().unwrap();
            opandt(x_slice, nc);
            make_non_zero_signed(&mut x);
        };
        cum_transform(cum_t1.view_mut());
        cum_transform(cum_t2.view_mut());
        opand_transform(input_0.slice_mut(s![run, ..]));
        let icumt_norm = |mut x: ArrayViewMut1<f64>| {
            let x_slice = x.as_slice_mut().unwrap();
            cumti(x_slice, nc);
            let s = 1e-100f64.max(x.sum());
            x /= s;
        };
        let v0 = input_0.slice(s![run, ..]);
        let set_input_distr = |mut op: ArrayViewMut1<f64>, t_other: ArrayView1<f64>| {
            ndarray::Zip::from(op.view_mut())
                .and(v0)
                .and(t_other)
                .for_each(|input, res, other| *input = *res * *other);
            make_non_zero_signed(&mut op);
            let op_slice = op.as_slice_mut().unwrap();
            opandt(op_slice, nc);
            let s = op.sum();
            op /= s;
            make_non_zero_signed(&mut op);
        };
        set_input_distr(input_1.slice_mut(s![run, ..]), cum_t2.view());
        set_input_distr(input_2.slice_mut(s![run, ..]), cum_t1.view());
        ndarray::Zip::from(input_0.slice_mut(s![run, ..]))
            .and(cum_t1.view())
            .and(cum_t2.view())
            .for_each(|res, in1, in2| *res = *in1 * *in2);
        icumt_norm(input_0.slice_mut(s![run, ..]));
    }
}

/// Run the belief propagation algorithm on the python representation of a factor graph.
pub fn run_bp(
    functions: &[Func],
    variables: &mut [Var],
    it: usize,
    // number of variable nodes in the graph
    edge: usize,
    // size of the field
    nc: usize,
    // number of copies in the graph (n_runs)
    n: usize,
    config: &crate::Config,
) -> Result<(), ()> {
    // Scratch array containing all the edge's messages.
    let mut edges: Vec<Array2<f64>> = vec![Array2::<f64>::ones((n, nc)); edge];

    // Mapping of each edge to its (function node id, position in function node).
    let mut vec_funcs_id: Vec<(usize, usize)> = vec![(0, 0); edge];
    // Mapping of each edge to its (variable node id, position in variable node).
    let mut vec_vars_id: Vec<(usize, usize)> = vec![(0, 0); edge];

    // map all python functions to rust ones + generate the mapping in vec_functs_id
    let functions_rust = functions;
    for (i, f) in functions.iter().enumerate() {
        f.neighboors.iter().enumerate().for_each(|(j, x)| {
            vec_funcs_id[*x] = (i, j);
        });
    }

    // map all python var to rust ones
    // generate the edge mapping in vec_vars_id
    // init the messages along the edges with initial distributions
    for (i, var) in variables.iter().enumerate() {
        var.neighboors.iter().enumerate().for_each(|(j, x)| {
            vec_vars_id[*x] = (i, j);
        });
        match &var.vartype {
            VarType::ProfilePara { distri_orig, .. }
            | VarType::ProfileSingle { distri_orig, .. } => var.neighboors.iter().for_each(|x| {
                let v = &mut edges[*x];
                let distri_orig = distri_orig.broadcast(v.shape()).unwrap();
                v.assign(&distri_orig);
            }),
            _ => {}
        }
    }

    let mut bp_iter = || {
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
        let mut edge_for_var: Vec<Vec<&mut Array2<f64>>> = variables
            .iter()
            .map(|f| {
                f.neighboors
                    .iter()
                    .map(|e| edge_opt_ref_mut[*e].take().unwrap())
                    .collect()
            })
            .collect();
        update_variables(&mut edge_for_var, variables);
    };

    crate::utils::with_progress(
        |it_cnt| {
            for _ in 0..it {
                bp_iter();
                it_cnt.inc(1);
            }
        },
        it.try_into().unwrap(),
        "Compute BP",
        config,
    );

    Ok(())
}
