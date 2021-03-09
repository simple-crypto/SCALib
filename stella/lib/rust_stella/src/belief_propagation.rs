use ndarray::{s, Array1, Array2, Array3, Axis};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::types::PyDict;

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
    neighboors: Vec<(usize, usize)>, // (id,offset)
    vartype: VarType,
    msg: Array3<f64>,
}

pub enum FuncType {
    AND,
    XOR,
    XORCST(Array1<u32>),
    LOOKUP(Array1<u32>),
}
pub struct Func {
    inputs: Vec<(usize, usize)>, // (id,offset)
    output: (usize, usize),
    neighboors: Vec<(usize, usize)>,
    functype: FuncType,
    msg: Array3<f64>,
}

pub fn to_var(function: &PyDict) -> Var {
    let neighboors: Vec<isize> = function.get_item("neighboors").unwrap().extract().unwrap();
    let inloop: bool = function.get_item("in_loop").unwrap().extract().unwrap();
    let offset: Vec<isize> = function.get_item("offset").unwrap().extract().unwrap();
    let is_profiled = function.contains("distri_orig").unwrap();
    let distri_current: PyReadonlyArray2<f64> =
        function.get_item("distri").unwrap().extract().unwrap();

    let neighboors: Vec<(usize, usize)> = neighboors
        .iter()
        .zip(offset.iter())
        .map(|(x, y)| (*x as usize, *y as usize))
        .collect();
    let msg: PyReadonlyArray3<f64> = function.get_item("msg").unwrap().extract().unwrap();

    let f: VarType;
    if inloop & is_profiled {
        let distri_orig: PyReadonlyArray2<f64> =
            function.get_item("distri_orig").unwrap().extract().unwrap();
        f = VarType::ProfilePara {
            distri_orig: distri_orig.as_array().to_owned(),
            distri_current: distri_current.as_array().to_owned(),
        };
    } else if inloop & !is_profiled {
        f = VarType::NotProfilePara {
            distri_current: distri_current.as_array().to_owned(),
        };
    } else if !inloop & is_profiled {
        let distri_orig: PyReadonlyArray2<f64> =
            function.get_item("distri_orig").unwrap().extract().unwrap();
        f = VarType::ProfilePara {
            distri_orig: distri_orig.as_array().to_owned(),
            distri_current: distri_current.as_array().to_owned(),
        };
    } else {
        f = VarType::NotProfilePara {
            distri_current: distri_current.as_array().to_owned(),
        };
    }
    // message to send
    let msg = msg.as_array();

    Var {
        neighboors: neighboors,
        vartype: f,
        msg: msg.to_owned(),
    }
}

pub fn to_func(function: &PyDict) -> Func {
    let neighboors: Vec<isize> = function.get_item("neighboors").unwrap().extract().unwrap();
    let func: usize = function.get_item("func").unwrap().extract().unwrap();
    let offset: Vec<isize> = function.get_item("offset").unwrap().extract().unwrap();

    let neighboors: Vec<(usize, usize)> = neighboors
        .iter()
        .zip(offset.iter())
        .map(|(x, y)| (*x as usize, *y as usize))
        .collect();
    let msg: PyReadonlyArray3<f64> = function.get_item("msg").unwrap().extract().unwrap();

    let mut inputs = neighboors.clone();
    let output = inputs.split_off(1)[0];

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
        inputs: inputs,
        output: output,
        neighboors: neighboors,
        functype: f,
        msg: msg.to_owned(),
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

pub fn update_variables(functions: &mut Vec<&mut Func>, variables: &mut Vec<&mut Var>) {
    variables.into_iter().for_each(|var| {
        // update the current distri
        match var.vartype {
            VarType::ProfilePara {
                distri_orig: distri_orig,
                distri_current: mut distri_current,
            } => {
                distri_current.assign(&distri_orig);
                distri_current.mapv_inplace(|x| f64::log2(x));

                var.neighboors.iter().for_each(|(id, offset)| {
                    let mut msg = &mut functions[*id].msg;
                    let mut msg = msg.slice_mut(s![.., *offset, ..]);
                    msg.mapv_inplace(|x| f64::log2(x));
                    distri_current += &msg;
                });
                distri_current -= &distri_current
                    .fold_axis(Axis(1), f64::MIN, |acc, x| acc.max(*x))
                    .insert_axis(Axis(1));
            }
            _ => (),
        }
        /*VarType::ProfileSingle {
            distri_orig: distri_orig,
            distri_current: mut distri_current,
        } => {
            distri_current.assign(&distri_orig);
            distri_current.mapv_inplace(|x| f64::log2(x));

            var.neighboors.iter().for_each(|(id, offset)| {
                let msg = &mut functions[*id].msg;
                let mut msg = msg.slice_mut(s![.., *offset, ..]);
                msg.mapv_inplace(|x| f64::log2(x));
                distri_current += &msg.sum_axis(Axis(0));
            });
            distri_current -= &distri_current
                .fold_axis(Axis(1), f64::MIN, |acc, x| acc.max(*x))
                .insert_axis(Axis(1));
        }
        VarType::NotProfilePara {
            distri_current: mut distri_current,
        } => {
            distri_current.fill(0.0);
            var.neighboors.iter().for_each(|(id, offset)| {
                let msg = &mut functions[*id].msg;
                let mut msg = msg.slice_mut(s![.., *offset, ..]);
                msg.mapv_inplace(|x| f64::log2(x));
                distri_current += &msg;
            });
            distri_current -= &distri_current
                .fold_axis(Axis(1), f64::MIN, |acc, x| acc.max(*x))
                .insert_axis(Axis(1));
        }
        VarType::NotProfileSingle {
            distri_current: mut distri_current,
        } => {
            distri_current.fill(0.0);
            var.neighboors.iter().for_each(|(id, offset)| {
                let msg = &mut functions[*id].msg;
                let mut msg = msg.slice_mut(s![.., *offset, ..]);
                msg.mapv_inplace(|x| f64::log2(x));
                distri_current += &msg.sum_axis(Axis(0));
            });

            distri_current -= &distri_current
                .fold_axis(Axis(1), f64::MIN, |acc, x| acc.max(*x))
                .insert_axis(Axis(1));
        }*/

        // send back the messages
        match &var.vartype {
            VarType::ProfilePara {
                distri_orig: _,
                distri_current: distri_current,
            } => {
                var.neighboors
                    .iter()
                    .zip(var.msg.axis_iter_mut(Axis(1)))
                    .for_each(|((id, offset), mut msg_out)| {
                        let msg_in = &functions[*id].msg;
                        let msg_in = msg_in.slice(s![.., *offset, ..]);
                        msg_out.assign(&(distri_current - &msg_in));
                        msg_out.mapv_inplace(|x| (2.0 as f64).powf(x));
                        msg_out /= &msg_out
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            .broadcast(msg_out.shape())
                            .unwrap();
                        msg_out.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                    });
            }
            _ => (),
        }

        /*
                    VarType::ProfileSingle {
                        distri_orig: _,
                        distri_current: distri_current,
                    } => {
                        var.neighboors
                            .iter()
                            .zip(var.msg.axis_iter_mut(Axis(1)))
                            .for_each(|((id, offset), mut msg_out)| {
                                let msg_in = &functions[*id].msg;
                                let msg_in = msg_in.slice(s![.., *offset, ..]);
                                let distri_current = distri_current.broadcast(msg_in.shape()).unwrap();
                                msg_out.assign(&(&distri_current - &msg_in));
                                msg_out.mapv_inplace(|x| (2.0 as f64).powf(x));
                                msg_out /= &msg_out
                                    .sum_axis(Axis(1))
                                    .insert_axis(Axis(1))
                                    .broadcast(msg_out.shape())
                                    .unwrap();
                                msg_out.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                            });
                    }
                    VarType::NotProfilePara {
                        distri_current: distri_current,
                    } => {
                        var.neighboors
                            .iter()
                            .zip(var.msg.axis_iter_mut(Axis(1)))
                            .for_each(|((id, offset), mut msg_out)| {
                                let msg_in = &functions[*id].msg;
                                let msg_in = msg_in.slice(s![.., *offset, ..]);
                                msg_out.assign(&(distri_current - &msg_in));
                                msg_out.mapv_inplace(|x| (2.0 as f64).powf(x));
                                msg_out /= &msg_out
                                    .sum_axis(Axis(1))
                                    .insert_axis(Axis(1))
                                    .broadcast(msg_out.shape())
                                    .unwrap();
                                msg_out.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                            });
                    }
                    VarType::NotProfileSingle {
                        distri_current: distri_current,
                    } => {
                        var.neighboors
                            .iter()
                            .zip(var.msg.axis_iter_mut(Axis(1)))
                            .for_each(|((id, offset), mut msg_out)| {
                                let msg_in = &functions[*id].msg;
                                let msg_in = msg_in.slice(s![.., *offset, ..]);
                                msg_out.assign(&(distri_current - &msg_in));
                                msg_out.mapv_inplace(|x| (2.0 as f64).powf(x));
                                msg_out /= &msg_out
                                    .sum_axis(Axis(1))
                                    .insert_axis(Axis(1))
                                    .broadcast(msg_out.shape())
                                    .unwrap();
                                msg_out.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                            });
                    }
                }
        */
    });
}

/*
pub fn update_functions(functions: &mut Vec<Func>, variables: &mut Vec<Var>) {
    functions.iter().for_each(|function| {
        let mut msg = function.msg;
        match function.functype{
            FuncType::AND =>{
                let mut output_msg = variables[function.neighboors[0].0].msg.slice_mut(s![..,function.neighboors[0].1,..]);
                let mut input1_msg = variables[function.neighboors[1].0].msg.slice_mut(s![..,function.neighboors[1].1,..]);
                let mut input2_msg = variables[function.neighboors[2].0].msg.slice_mut(s![..,function.neighboors[2].1,..]);
                let nc = input1_msg.shape()[1];
                function.msg.fill(0.0);
                function.msg.outer_iter_mut()
                    .zip(input1_msg.outer_iter_mut())
                    .zip(input2_msg.outer_iter_mut())
                    .zip(output_msg.outer_iter_mut())
                    .for_each(
                        |(((mut msg, mut input1_msg), mut input2_msg), mut output_msg)| {
                            let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                            let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                            let output_msg_s = output_msg.as_slice_mut().unwrap();

                            let mut msg = msg.slice_mut(s![0..3, ..]);
                            let msg_s = msg.as_slice_mut().unwrap();
                            for i1 in 0..nc {
                                for i2 in 0..nc {
                                    let o: usize = i1 & i2;
                                    // input 1
                                    msg_s[nc + i1] += input2_msg_s[i2] * output_msg_s[o];
                                    // input 2
                                    msg_s[2 * nc + i2] += input1_msg_s[i1] * output_msg_s[o];
                                    // out
                                    msg_s[o] += input1_msg_s[i1] * input2_msg_s[i2];
                                }
                            }
                        });
            }
            FuncType::XOR =>{
                let inputs_msg : Vec<ArrayViewMut2<f64>> = function.neighboors.iter().map(|(id,offset)|{
                    variables[*id].msg.slice_mut(s![..,*offset,..])
                }).collect();
                let output_msg = inputs_msg.split_off(1)[0];
                let nc = output_msg.shape()[1];
                if inputs_msg.len() == 2{
                    xor_2(inputs_msg,output_msg,function.msg.view_mut(),nc);
                }else{
                    xor_3(inputs_msg,output_msg,function.msg.view_mut(),nc);
                }
            }
            FuncType::XORCST(values) => {
                let mut output_msg = variables[function.neighboors[0].0].msg.slice_mut(s![..,function.neighboors[0].1,..]);
                let mut input1_msg = variables[function.neighboors[1].0].msg.slice_mut(s![..,function.neighboors[1].1,..]);
                let nc = input1_msg.shape()[1];
                function.msg.fill(0.0);
                function.msg.outer_iter_mut()
                    .zip(input1_msg.outer_iter_mut())
                    .zip(values.iter())
                    .zip(output_msg.outer_iter_mut())
                    .for_each(
                        |(((mut msg, mut input1_msg), values), mut output_msg)| {
                            let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                            let output_msg_s = output_msg.as_slice_mut().unwrap();

                            let mut msg = msg.slice_mut(s![0..2, ..]);
                            let msg_s = msg.as_slice_mut().unwrap();
                            for i1 in 0..nc {
                                    let o: usize = ((i1 as u32) ^ *values) as usize;
                                    // output
                                    msg_s[o] = input1_msg_s[i1];
                                    msg_s[nc + i1] = output_msg_s[o];
                            }
                        });
            }
            FuncType::LOOKUP(table) => {
                let mut output_msg = variables[function.neighboors[0].0].msg.slice_mut(s![..,function.neighboors[0].1,..]);
                let mut input1_msg = variables[function.neighboors[1].0].msg.slice_mut(s![..,function.neighboors[1].1,..]);
                let nc = input1_msg.shape()[1];
                let table = table.as_slice().unwrap();
                function.msg.fill(0.0);
                function.msg.outer_iter_mut()
                    .zip(input1_msg.outer_iter_mut())
                    .zip(output_msg.outer_iter_mut())
                    .for_each(
                        |((mut msg, mut input1_msg), mut output_msg)| {
                            let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                            let output_msg_s = output_msg.as_slice_mut().unwrap();

                            let mut msg = msg.slice_mut(s![0..2, ..]);
                            let msg_s = msg.as_slice_mut().unwrap();
                            for i1 in 0..nc {
                                    let o: usize = table[i1] as usize;
                                    // output
                                    msg_s[o] = input1_msg_s[i1];
                                    msg_s[nc + i1] = output_msg_s[o];
                            }
                        });
            }
            _ => (),
        }
    });
}
*/
/*
fn xor_4(
    inputs_v: Vec<&PyDict>,
    offset: Vec<isize>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input1_msg_s = input_msg.slice_mut(s![.., offset[1], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[1].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input2_msg_s = input_msg.slice_mut(s![.., offset[2], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[2].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input3_msg_s = input_msg.slice_mut(s![.., offset[3], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[3].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input4_msg_s = input_msg.slice_mut(s![.., offset[4], ..]);

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
        .zip(input3_msg_s.outer_iter_mut())
        .zip(input4_msg_s.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |(
                ((((mut msg, mut input1_msg), mut input2_msg), mut input3_msg), mut input4_msg),
                mut output_msg,
            )| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let input3_msg_s = input3_msg.as_slice_mut().unwrap();
                let input4_msg_s = input4_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(input3_msg_s, nc);
                fwht(input4_msg_s, nc);
                fwht(output_msg_s, nc);
                input1_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input2_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input3_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input4_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                output_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });

                for i in 0..nc {
                    tmp_all_s[i] = input1_msg_s[i]
                        * input2_msg_s[i]
                        * input3_msg_s[i]
                        * input4_msg_s[i]
                        * output_msg_s[i];
                }
                let mut vecs = Vec::with_capacity(5);
                vecs.push(output_msg_s);
                vecs.push(input1_msg_s);
                vecs.push(input2_msg_s);
                vecs.push(input3_msg_s);
                vecs.push(input4_msg_s);

                for (i, msg_s) in vecs.iter().enumerate() {
                    // message to the output
                    let mut tmp = msg.slice_mut(s![i, ..]);
                    let tmp_s = tmp.as_slice_mut().unwrap();
                    for i in 0..nc {
                        tmp_s[i] = tmp_all_s[i] / msg_s[i];
                    }
                    fwht(tmp_s, nc);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                }
            },
        );
}*/

/*
fn xor_2(
    inputs_v: Vec<&mut ArrayViewMut2<f64>>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    let input1_msg = &inputs_v[0];
    let input2_msg = &inputs_v[1];
    msg.outer_iter_mut()
        .zip(input1_msg.outer_iter_mut())
        .zip(input2_msg.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |(((mut msg, mut input1_msg), mut input2_msg), mut output_msg)| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(output_msg_s, nc);

                // message to the output
                let mut tmp = msg.slice_mut(s![0, ..]);
                tmp.assign(&(&input2_msg * &input1_msg));
                let tmp_s = tmp.as_slice_mut().unwrap();
                fwht(tmp_s, nc);
                let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                tmp_s.iter_mut().for_each(|x| *x /= s);

                // message to the input 1
                let mut tmp = msg.slice_mut(s![1, ..]);
                tmp.assign(&(&input2_msg * &output_msg));
                let tmp_s = tmp.as_slice_mut().unwrap();
                fwht(tmp_s, nc);
                let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                tmp_s.iter_mut().for_each(|x| *x /= s);

                // message to the input 2
                let mut tmp = msg.slice_mut(s![2, ..]);
                tmp.assign(&(&input1_msg * &output_msg));
                let tmp_s = tmp.as_slice_mut().unwrap();
                fwht(tmp_s, nc);
                let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                tmp_s.iter_mut().for_each(|x| *x /= s);
            },
        );
}

fn xor_3(
    inputs_v: Vec<&mut ArrayViewMut2<f64>>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    let input1_msg = inputs_v[0];
    let input2_msg = inputs_v[1];
    let input3_msg = inputs_v[2];

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg.outer_iter_mut())
        .zip(input2_msg.outer_iter_mut())
        .zip(input3_msg.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |((((mut msg, mut input1_msg), mut input2_msg), mut input3_msg), mut output_msg)| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let input3_msg_s = input3_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(input3_msg_s, nc);
                fwht(output_msg_s, nc);
                input1_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input2_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input3_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                output_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });

                for i in 0..nc {
                    tmp_all_s[i] =
                        input1_msg_s[i] * input2_msg_s[i] * input3_msg_s[i] * output_msg_s[i];
                }
                let mut vecs = Vec::with_capacity(4);
                vecs.push(output_msg_s);
                vecs.push(input1_msg_s);
                vecs.push(input2_msg_s);
                vecs.push(input3_msg_s);

                for (i, msg_s) in vecs.iter().enumerate() {
                    // message to the output
                    let mut tmp = msg.slice_mut(s![i, ..]);
                    let tmp_s = tmp.as_slice_mut().unwrap();
                    for i in 0..nc {
                        tmp_s[i] = tmp_all_s[i] / msg_s[i];
                    }
                    fwht(tmp_s, nc);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                }
            },
        );
}
*/

/*
fn xor_5(
    inputs_v: Vec<&PyDict>,
    offset: Vec<isize>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input1_msg_s = input_msg.slice_mut(s![.., offset[1], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[1].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input2_msg_s = input_msg.slice_mut(s![.., offset[2], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[2].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input3_msg_s = input_msg.slice_mut(s![.., offset[3], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[3].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input4_msg_s = input_msg.slice_mut(s![.., offset[4], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[4].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input5_msg_s = input_msg.slice_mut(s![.., offset[5], ..]);

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
        .zip(input3_msg_s.outer_iter_mut())
        .zip(input4_msg_s.outer_iter_mut())
        .zip(input5_msg_s.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |(
                (
                    ((((mut msg, mut input1_msg), mut input2_msg), mut input3_msg), mut input4_msg),
                    mut input5_msg,
                ),
                mut output_msg,
            )| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let input3_msg_s = input3_msg.as_slice_mut().unwrap();
                let input4_msg_s = input4_msg.as_slice_mut().unwrap();
                let input5_msg_s = input5_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(input3_msg_s, nc);
                fwht(input4_msg_s, nc);
                fwht(input5_msg_s, nc);
                fwht(output_msg_s, nc);
                input1_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input2_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input3_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input4_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input5_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                output_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });

                for i in 0..nc {
                    tmp_all_s[i] = input1_msg_s[i]
                        * input2_msg_s[i]
                        * input3_msg_s[i]
                        * input4_msg_s[i]
                        * input5_msg_s[i]
                        * output_msg_s[i];
                }
                let mut vecs = Vec::with_capacity(6);
                vecs.push(output_msg_s);
                vecs.push(input1_msg_s);
                vecs.push(input2_msg_s);
                vecs.push(input3_msg_s);
                vecs.push(input4_msg_s);
                vecs.push(input5_msg_s);

                for (i, msg_s) in vecs.iter().enumerate() {
                    // message to the output
                    let mut tmp = msg.slice_mut(s![i, ..]);
                    let tmp_s = tmp.as_slice_mut().unwrap();
                    for i in 0..nc {
                        tmp_s[i] = tmp_all_s[i] / msg_s[i];
                    }
                    fwht(tmp_s, nc);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                }
            },
        );
}

fn xor_6(
    inputs_v: Vec<&PyDict>,
    offset: Vec<isize>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input1_msg_s = input_msg.slice_mut(s![.., offset[1], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[1].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input2_msg_s = input_msg.slice_mut(s![.., offset[2], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[2].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input3_msg_s = input_msg.slice_mut(s![.., offset[3], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[3].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input4_msg_s = input_msg.slice_mut(s![.., offset[4], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[4].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input5_msg_s = input_msg.slice_mut(s![.., offset[5], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[5].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input6_msg_s = input_msg.slice_mut(s![.., offset[6], ..]);

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
        .zip(input3_msg_s.outer_iter_mut())
        .zip(input4_msg_s.outer_iter_mut())
        .zip(input5_msg_s.outer_iter_mut())
        .zip(input6_msg_s.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |(
                (
                    (
                        (
                            (((mut msg, mut input1_msg), mut input2_msg), mut input3_msg),
                            mut input4_msg,
                        ),
                        mut input5_msg,
                    ),
                    mut input6_msg,
                ),
                mut output_msg,
            )| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let input3_msg_s = input3_msg.as_slice_mut().unwrap();
                let input4_msg_s = input4_msg.as_slice_mut().unwrap();
                let input5_msg_s = input5_msg.as_slice_mut().unwrap();
                let input6_msg_s = input6_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(input3_msg_s, nc);
                fwht(input4_msg_s, nc);
                fwht(input5_msg_s, nc);
                fwht(input6_msg_s, nc);
                fwht(output_msg_s, nc);
                input1_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input2_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input3_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input4_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input5_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input6_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                output_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });

                for i in 0..nc {
                    tmp_all_s[i] = input1_msg_s[i]
                        * input2_msg_s[i]
                        * input3_msg_s[i]
                        * input4_msg_s[i]
                        * input5_msg_s[i]
                        * input6_msg_s[i]
                        * output_msg_s[i];
                }
                let mut vecs = Vec::with_capacity(7);
                vecs.push(output_msg_s);
                vecs.push(input1_msg_s);
                vecs.push(input2_msg_s);
                vecs.push(input3_msg_s);
                vecs.push(input4_msg_s);
                vecs.push(input5_msg_s);
                vecs.push(input6_msg_s);

                for (i, msg_s) in vecs.iter().enumerate() {
                    // message to the output
                    let mut tmp = msg.slice_mut(s![i, ..]);
                    let tmp_s = tmp.as_slice_mut().unwrap();
                    for i in 0..nc {
                        tmp_s[i] = tmp_all_s[i] / msg_s[i];
                    }
                    fwht(tmp_s, nc);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                }
            },
        );
}

fn xor_7(
    inputs_v: Vec<&PyDict>,
    offset: Vec<isize>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input1_msg_s = input_msg.slice_mut(s![.., offset[1], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[1].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input2_msg_s = input_msg.slice_mut(s![.., offset[2], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[2].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input3_msg_s = input_msg.slice_mut(s![.., offset[3], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[3].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input4_msg_s = input_msg.slice_mut(s![.., offset[4], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[4].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input5_msg_s = input_msg.slice_mut(s![.., offset[5], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[5].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input6_msg_s = input_msg.slice_mut(s![.., offset[6], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[6].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input7_msg_s = input_msg.slice_mut(s![.., offset[7], ..]);

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
        .zip(input3_msg_s.outer_iter_mut())
        .zip(input4_msg_s.outer_iter_mut())
        .zip(input5_msg_s.outer_iter_mut())
        .zip(input6_msg_s.outer_iter_mut())
        .zip(input7_msg_s.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |(
                (
                    (
                        (
                            (
                                (((mut msg, mut input1_msg), mut input2_msg), mut input3_msg),
                                mut input4_msg,
                            ),
                            mut input5_msg,
                        ),
                        mut input6_msg,
                    ),
                    mut input7_msg,
                ),
                mut output_msg,
            )| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let input3_msg_s = input3_msg.as_slice_mut().unwrap();
                let input4_msg_s = input4_msg.as_slice_mut().unwrap();
                let input5_msg_s = input5_msg.as_slice_mut().unwrap();
                let input6_msg_s = input6_msg.as_slice_mut().unwrap();
                let input7_msg_s = input7_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(input3_msg_s, nc);
                fwht(input4_msg_s, nc);
                fwht(input5_msg_s, nc);
                fwht(input6_msg_s, nc);
                fwht(input7_msg_s, nc);
                fwht(output_msg_s, nc);
                input1_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input2_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input3_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input4_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input5_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input6_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input7_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                output_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });

                for i in 0..nc {
                    tmp_all_s[i] = input1_msg_s[i]
                        * input2_msg_s[i]
                        * input3_msg_s[i]
                        * input4_msg_s[i]
                        * input5_msg_s[i]
                        * input6_msg_s[i]
                        * input7_msg_s[i]
                        * output_msg_s[i];
                }
                let mut vecs = Vec::with_capacity(8);
                vecs.push(output_msg_s);
                vecs.push(input1_msg_s);
                vecs.push(input2_msg_s);
                vecs.push(input3_msg_s);
                vecs.push(input4_msg_s);
                vecs.push(input5_msg_s);
                vecs.push(input6_msg_s);
                vecs.push(input7_msg_s);

                for (i, msg_s) in vecs.iter().enumerate() {
                    // message to the output
                    let mut tmp = msg.slice_mut(s![i, ..]);
                    let tmp_s = tmp.as_slice_mut().unwrap();
                    for i in 0..nc {
                        tmp_s[i] = tmp_all_s[i] / msg_s[i];
                    }
                    fwht(tmp_s, nc);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                }
            },
        );
}

fn xor_8(
    inputs_v: Vec<&PyDict>,
    offset: Vec<isize>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input1_msg_s = input_msg.slice_mut(s![.., offset[1], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[1].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input2_msg_s = input_msg.slice_mut(s![.., offset[2], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[2].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input3_msg_s = input_msg.slice_mut(s![.., offset[3], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[3].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input4_msg_s = input_msg.slice_mut(s![.., offset[4], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[4].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input5_msg_s = input_msg.slice_mut(s![.., offset[5], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[5].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input6_msg_s = input_msg.slice_mut(s![.., offset[6], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[6].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input7_msg_s = input_msg.slice_mut(s![.., offset[7], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[7].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input8_msg_s = input_msg.slice_mut(s![.., offset[8], ..]);

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
        .zip(input3_msg_s.outer_iter_mut())
        .zip(input4_msg_s.outer_iter_mut())
        .zip(input5_msg_s.outer_iter_mut())
        .zip(input6_msg_s.outer_iter_mut())
        .zip(input7_msg_s.outer_iter_mut())
        .zip(input8_msg_s.outer_iter_mut())
        .zip(output_msg.outer_iter_mut())
        .for_each(
            |(
                (
                    (
                        (
                            (
                                (
                                    (((mut msg, mut input1_msg), mut input2_msg), mut input3_msg),
                                    mut input4_msg,
                                ),
                                mut input5_msg,
                            ),
                            mut input6_msg,
                        ),
                        mut input7_msg,
                    ),
                    mut input8_msg,
                ),
                mut output_msg,
            )| {
                let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                let input3_msg_s = input3_msg.as_slice_mut().unwrap();
                let input4_msg_s = input4_msg.as_slice_mut().unwrap();
                let input5_msg_s = input5_msg.as_slice_mut().unwrap();
                let input6_msg_s = input6_msg.as_slice_mut().unwrap();
                let input7_msg_s = input7_msg.as_slice_mut().unwrap();
                let input8_msg_s = input8_msg.as_slice_mut().unwrap();
                let output_msg_s = output_msg.as_slice_mut().unwrap();
                fwht(input1_msg_s, nc);
                fwht(input2_msg_s, nc);
                fwht(input3_msg_s, nc);
                fwht(input4_msg_s, nc);
                fwht(input5_msg_s, nc);
                fwht(input6_msg_s, nc);
                fwht(input7_msg_s, nc);
                fwht(input8_msg_s, nc);
                fwht(output_msg_s, nc);
                input1_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input2_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input3_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input4_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input5_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input6_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input7_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                input8_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });
                output_msg_s
                    .iter_mut()
                    .for_each(|x| *x = if f64::abs(*x) < 1E-10 { 1E-10 } else { *x });

                for i in 0..nc {
                    tmp_all_s[i] = input1_msg_s[i]
                        * input2_msg_s[i]
                        * input3_msg_s[i]
                        * input4_msg_s[i]
                        * input5_msg_s[i]
                        * input6_msg_s[i]
                        * input7_msg_s[i]
                        * input8_msg_s[i]
                        * output_msg_s[i];
                }
                let mut vecs = Vec::with_capacity(9);
                vecs.push(output_msg_s);
                vecs.push(input1_msg_s);
                vecs.push(input2_msg_s);
                vecs.push(input3_msg_s);
                vecs.push(input4_msg_s);
                vecs.push(input5_msg_s);
                vecs.push(input6_msg_s);
                vecs.push(input7_msg_s);
                vecs.push(input8_msg_s);

                for (i, msg_s) in vecs.iter().enumerate() {
                    // message to the output
                    let mut tmp = msg.slice_mut(s![i, ..]);
                    let tmp_s = tmp.as_slice_mut().unwrap();
                    for i in 0..nc {
                        tmp_s[i] = tmp_all_s[i] / msg_s[i];
                    }
                    fwht(tmp_s, nc);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                }
            },
        );
}*/
