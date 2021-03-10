use ndarray::{
    s, Array1, Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::types::PyDict;
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
    let is_profiled = function.contains("distri_orig").unwrap();
    let distri_current: PyReadonlyArray2<f64> =
        function.get_item("distri").unwrap().extract().unwrap();

    let neighboors: Vec<usize> = neighboors.iter().map(|x| *x as usize).collect();
    let f: VarType;
    if inloop & is_profiled {
        let distri_orig: PyReadonlyArray2<f64> =
            function.get_item("distri_orig").unwrap().extract().unwrap();
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
            function.get_item("distri_orig").unwrap().extract().unwrap();
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
        .iter_mut()
        .zip(vertex.iter_mut())
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
        .iter_mut()
        .zip(vertex.iter_mut())
        .for_each(|(function, mut vertex)| {
            match &mut function.functype {
                FuncType::AND => {
                    let input2_msg = vertex.pop().unwrap();
                    let input1_msg = vertex.pop().unwrap();
                    let output_msg = vertex.pop().unwrap();
                    let nc = input1_msg.shape()[1];
                    input1_msg
                        .outer_iter_mut()
                        .zip(input2_msg.outer_iter_mut())
                        .zip(output_msg.outer_iter_mut())
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
                    let mut input_msg = vertex.split_off(1);
                    let mut output_msg = vertex.pop().unwrap();
                    let nc = output_msg.shape()[1];
                    xors(&mut input_msg, output_msg, nc);
                }
                FuncType::XORCST(values) => {
                    let input1_msg = vertex.pop().unwrap();
                    let output_msg = vertex.pop().unwrap();
                    let nc = input1_msg.shape()[1];
                    input1_msg
                        .outer_iter_mut()
                        .zip(output_msg.outer_iter_mut())
                        .zip(values.iter())
                        .for_each(|((mut input1_msg, mut output_msg), value)| {
                            let input1_msg_o = input1_msg.to_owned();
                            let output_msg_o = output_msg.to_owned();
                            let input1_msg_s = input1_msg_o.as_slice().unwrap();
                            let output_msg_s = output_msg_o.as_slice().unwrap();

                            input1_msg.fill(0.0);
                            output_msg.fill(0.0);
                            let input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                            let output_msg_s_mut = output_msg.as_slice_mut().unwrap();

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
                        .zip(output_msg.outer_iter_mut())
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

fn xors(inputs: &mut Vec<&mut Array2<f64>>, output: &mut Array2<f64>, nc: usize) {
    output
        .outer_iter_mut()
        .enumerate()
        .for_each(|(i, mut output)| {
            let mut acc = Array1::<f64>::zeros(nc);
            // set the output
            let output_s = output.as_slice_mut().unwrap();
            fwht(output_s, nc);
            output_s
                .iter_mut()
                .for_each(|x| *x = if f64::abs(*x) == 0.0 { 1E-50 } else { *x });
            acc.assign(&output);
            
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
            output.zip_mut_with(&acc, |x, y| *x = *y / *x);
            let output_fwt_s = output.as_slice_mut().unwrap();
            fwht(output_fwt_s, nc);
            let s = output.iter().fold(0.0, |acc, x| acc + x.max(1E-50));
            output
                .iter_mut()
                .for_each(|x| *x = (x.max(1E-50) / s).max(1E-50));
        });
}
