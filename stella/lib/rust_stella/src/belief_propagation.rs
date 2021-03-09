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
    neighboors: Vec<usize>,
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
    let offset: Vec<isize> = function.get_item("offset").unwrap().extract().unwrap();

    let neighboors: Vec<usize> = neighboors.iter().map(|x| *x as usize).collect();
    let msg: PyReadonlyArray3<f64> = function.get_item("msg").unwrap().extract().unwrap();

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

    // message to send
    let msg = msg.as_array();

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

pub fn update_variables(vertex: &mut Vec<Array2<f64>>, variables: &mut Vec<Var>) {
    variables.iter_mut().for_each(|var| {
        // update the current distri
        match &mut var.vartype {
            VarType::ProfilePara {
                distri_orig,
                distri_current,
            } => {
                distri_current.assign(&distri_orig);
                var.neighboors.iter().for_each(|id| {
                    let msg = &mut vertex[*id];
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
                var.neighboors.iter().for_each(|id| {
                    let msg = &mut vertex[*id];
                    msg.outer_iter().for_each(|msg| {
                        *distri_current *= &msg;
                        *distri_current /= distri_current.fold(0.0, |acc, x| acc + *x);
                    });
                });
            }
            VarType::NotProfilePara { distri_current } => {
                distri_current.fill(1.0);
                var.neighboors.iter().for_each(|id| {
                    let msg = &mut vertex[*id];
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
                var.neighboors.iter().for_each(|id| {
                    let msg = &mut vertex[*id];
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
                var.neighboors.iter().for_each(|id| {
                    let mut msg = &mut vertex[*id];
                    msg.zip_mut_with(distri_current, |msg, distri| *msg = *distri / *msg);
                    *msg /= &msg
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
                var.neighboors.iter().for_each(|id| {
                    let mut msg = &mut vertex[*id];
                    let distri_current = distri_current.broadcast(msg.shape()).unwrap();
                    msg.zip_mut_with(&distri_current, |msg, distri| *msg = *distri / *msg);
                    *msg /= &msg
                        .sum_axis(Axis(1))
                        .insert_axis(Axis(1))
                        .broadcast(msg.shape())
                        .unwrap();
                    msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                });
            }
            VarType::NotProfilePara { distri_current } => {
                var.neighboors.iter().for_each(|id| {
                    let mut msg = &mut vertex[*id];
                    msg.zip_mut_with(distri_current, |msg, distri| *msg = *distri / *msg);
                    *msg /= &msg
                        .sum_axis(Axis(1))
                        .insert_axis(Axis(1))
                        .broadcast(msg.shape())
                        .unwrap();
                    msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                });
            }
            VarType::NotProfileSingle { distri_current } => {
                var.neighboors.iter().for_each(|id| {
                    let mut msg = &mut vertex[*id];
                    let distri_current = distri_current.broadcast(msg.shape()).unwrap();
                    msg.zip_mut_with(&distri_current, |msg, distri| *msg = *distri / *msg);
                    *msg /= &msg
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

pub fn update_functions(functions: &mut Vec<Func>, vertex: &mut Vec<Array2<f64>>) {
    functions.iter_mut().for_each(|function| {
        match &mut function.functype {
            FuncType::AND => {
                let mut output_msg = &mut vertex[function.neighboors[0]];
                let mut input1_msg = &mut vertex[function.neighboors[1]];
                let mut input2_msg = &mut vertex[function.neighboors[2]];
                let nc = input1_msg.shape()[1];
                input1_msg
                    .outer_iter_mut()
                    .zip(input2_msg.outer_iter_mut())
                    .zip(output_msg.outer_iter_mut())
                    .for_each(|((mut input1_msg, mut input2_msg), mut output_msg)| {
                        let input1_msg_s = input1_msg.to_owned().as_slice().unwrap();
                        let input2_msg_s = input2_msg.to_owned().as_slice().unwrap();
                        let output_msg_s = output_msg.to_owned().as_slice().unwrap();

                        input1_msg.fill(0.0);
                        input2_msg.fill(0.0);
                        output_msg.fill(0.0);
                        let mut input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                        let mut input2_msg_s_mut = input2_msg.as_slice_mut().unwrap();
                        let mut output_msg_s_mut = output_msg.as_slice_mut().unwrap();

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
                let mut output_msg: Vec<ArrayViewMut2<f64>> = function
                    .neighboors
                    .iter()
                    .map(|id| vertex[*id].view_mut())
                    .collect();
                let inputs_msg = output_msg.split_off(1);
                let output_msg = output_msg[0];
                let nc = output_msg.shape()[1];
                xors(&inputs_msg, output_msg, nc);
            }
            FuncType::XORCST(values) => {
                let mut output_msg = vertex[function.neighboors[0]];
                let mut input1_msg = vertex[function.neighboors[1]];
                let nc = input1_msg.shape()[1];
                input1_msg
                    .outer_iter_mut()
                    .zip(output_msg.outer_iter_mut())
                    .zip(values.iter())
                    .for_each(|((mut input1_msg, mut output_msg), value)| {
                        let input1_msg_s = input1_msg.to_owned().as_slice().unwrap();
                        let output_msg_s = output_msg.to_owned().as_slice().unwrap();

                        input1_msg.fill(0.0);
                        output_msg.fill(0.0);
                        let mut input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                        let mut output_msg_s_mut = output_msg.as_slice_mut().unwrap();

                        for i1 in 0..nc {
                            let o: usize = ((i1 as u32) ^ value) as usize;
                            input1_msg_s_mut[i1] += output_msg_s[o];
                            output_msg_s_mut[o] += input1_msg_s[i1];
                        }
                    });
            }
            FuncType::LOOKUP(table) => {
                let mut output_msg = vertex[function.neighboors[0]];
                let mut input1_msg = vertex[function.neighboors[1]];
                let nc = input1_msg.shape()[1];
                let table = table.as_slice().unwrap();
                input1_msg
                    .outer_iter_mut()
                    .zip(output_msg.outer_iter_mut())
                    .for_each(|(mut input1_msg, mut output_msg)| {
                        let input1_msg_s = input1_msg.to_owned().as_slice().unwrap();
                        let output_msg_s = output_msg.to_owned().as_slice().unwrap();

                        input1_msg.fill(0.0);
                        output_msg.fill(0.0);
                        let mut input1_msg_s_mut = input1_msg.as_slice_mut().unwrap();
                        let mut output_msg_s_mut = output_msg.as_slice_mut().unwrap();

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

fn xor_2(
    inputs_v: &Vec<ArrayView2<f64>>,
    output_msg: &ArrayView2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    let input1_msg = &inputs_v[0];
    let input2_msg = &inputs_v[1];
    msg.outer_iter_mut()
        .zip(input1_msg.outer_iter())
        .zip(input2_msg.outer_iter())
        .zip(output_msg.outer_iter())
        .for_each(|(((mut msg, input1_msg), input2_msg), output_msg)| {
            let mut input1_msg = input1_msg.to_owned();
            let mut input2_msg = input2_msg.to_owned();
            let mut output_msg = output_msg.to_owned();
            let input1_msg_s = input1_msg.as_slice_mut().unwrap();
            let input2_msg_s = input2_msg.as_slice_mut().unwrap();
            let output_msg_s = output_msg.as_slice_mut().unwrap();
            fwht(input1_msg_s, nc);
            fwht(input2_msg_s, nc);
            fwht(output_msg_s, nc);
            // message to the output
            let mut tmp = msg.slice_mut(s![0, ..]);
            tmp.assign(&input2_msg);
            tmp *= &input1_msg;
            let tmp_s = tmp.as_slice_mut().unwrap();
            fwht(tmp_s, nc);
            let s = tmp_s.iter().fold(0.0, |acc, x| acc + f64::abs(*x));
            tmp_s.iter_mut().for_each(|x| *x = (*x / s).max(1E-50));

            // message to the input 1
            let mut tmp = msg.slice_mut(s![1, ..]);
            tmp.assign(&input2_msg);
            tmp *= &output_msg;
            let tmp_s = tmp.as_slice_mut().unwrap();
            fwht(tmp_s, nc);
            let s = tmp_s.iter().fold(0.0, |acc, x| acc + f64::abs(*x));
            tmp_s.iter_mut().for_each(|x| *x = (*x / s).max(1E-50));

            // message to the input 2
            let mut tmp = msg.slice_mut(s![2, ..]);
            tmp.assign(&input1_msg);
            tmp *= &output_msg;
            let tmp_s = tmp.as_slice_mut().unwrap();
            fwht(tmp_s, nc);
            let s = tmp_s.iter().fold(0.0, |acc, x| acc + f64::abs(*x));
            tmp_s.iter_mut().for_each(|x| *x = (*x / s).max(1E-50));
        });
}

fn xors(inputs: &Vec<ArrayViewMut2<f64>>, output: ArrayViewMut2<f64>, nc: usize) {
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
            inputs.iter_mut().for_each(|mut input| {
                let mut input_fwt_s = input.as_slice_mut().unwrap();
                fwht(input_fwt_s, nc);
                input_fwt_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) == 0.0 { 1E-50 } else { *x });
                acc.zip_mut_with(input, |x, y| *x = *x / y);
                acc /= acc.sum();
            });
            inputs.iter_mut().for_each(|mut input| {
                input.zip_mut_with(&acc, |x, y| *x = *y / *x);
                let mut input_fwt_s = input.as_slice_mut().unwrap();
                fwht(input_fwt_s, nc);
                let s = input.iter().fold(0.0, |acc, x| acc + x.max(1E-50));
                input
                    .iter_mut()
                    .for_each(|x| *x = (x.max(1E-50) / s).max(1E-50));
            });
            output.zip_mut_with(&acc, |x, y| *x = *y / *x);
            let mut output_fwt_s = output.as_slice_mut().unwrap();
            fwht(output_fwt_s, nc);
            let s = output.iter().fold(0.0, |acc, x| acc + x.max(1E-50));
            output
                .iter_mut()
                .for_each(|x| *x = (x.max(1E-50) / s).max(1E-50));
        });
}
