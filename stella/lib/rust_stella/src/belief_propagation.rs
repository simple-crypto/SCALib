use ndarray::{s, Array1, ArrayViewMut2, ArrayViewMut3, Axis};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::types::{PyDict, PyList};

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

pub fn update_variables(functions: &PyList, variables: &PyList) {
    variables.iter().for_each(|var| {
        let var = var.downcast::<PyDict>().unwrap();
        let in_loop: bool = var.get_item("in_loop").unwrap().extract().unwrap();
        if var.contains("distri").unwrap() {
            // get the current distri
            let distri: &PyArray2<f64> = var.get_item("distri").unwrap().extract().unwrap();
            let mut distri = unsafe { distri.as_array_mut() };

            // get all the messages to sent (n,n_neighboors,nc)
            let msg: &PyArray3<f64> = var.get_item("msg").unwrap().extract().unwrap();
            let mut msg = unsafe { msg.as_array_mut() };

            // get offset in all the neighboors and list all the neighboors
            let offset: Vec<isize> = var.get_item("offset").unwrap().extract().unwrap();
            let neighboors: Vec<isize> = var.get_item("neighboors").unwrap().extract().unwrap();
            let neighboors: Vec<&PyDict> = neighboors
                .iter()
                .map(|x| functions.get_item(*x).extract().unwrap())
                .collect();

            if var.contains("distri_orig").unwrap() {
                let distri_orig: PyReadonlyArray2<f64> =
                    var.get_item("distri_orig").unwrap().extract().unwrap();
                let distri_orig = distri_orig.as_array();
                distri.assign(&distri_orig);
                distri.mapv_inplace(|x| f64::log2(x));
            } else {
                distri.fill(0.0);
            }

            offset
                .iter()
                .zip(neighboors.iter())
                .for_each(|(offset, neighboor)| {
                    let msg_in: &PyArray3<f64> =
                        neighboor.get_item("msg").unwrap().extract().unwrap();

                    let mut msg_in = unsafe { msg_in.as_array_mut() };
                    let mut msg_in = msg_in.slice_mut(s![.., *offset, ..]);
                    msg_in.mapv_inplace(|x| f64::log2(x));
                    let in_loop_neighboor: bool =
                        neighboor.get_item("in_loop").unwrap().extract().unwrap();

                    if (in_loop == false) & (in_loop_neighboor == true) {
                        distri += &msg_in.sum_axis(Axis(0));
                    } else {
                        distri += &msg_in;
                    }
                });

            let max = distri
                .fold_axis(Axis(1), f64::MIN, |acc, x| acc.max(*x))
                .insert_axis(Axis(1));
            distri -= &max;
            offset.iter().zip(neighboors.iter()).enumerate().for_each(
                |(i, (offset, neighboor))| {
                    let msg_in: PyReadonlyArray3<f64> =
                        neighboor.get_item("msg").unwrap().extract().unwrap();
                    let in_loop_neighboor: bool =
                        neighboor.get_item("in_loop").unwrap().extract().unwrap();

                    let msg_in = msg_in.as_array();
                    let msg_in = msg_in.slice(s![.., *offset, ..]);
                    let mut msg_out = msg.slice_mut(s![.., i, ..]);
                    if (in_loop == false) & (in_loop_neighboor == true) {
                        let distri = distri.broadcast(msg_in.shape()).unwrap();
                        msg_out.assign(&(&distri - &msg_in));
                    } else {
                        msg_out.assign(&(&distri - &msg_in));
                    }
                    msg_out.mapv_inplace(|x| (2.0 as f64).powf(x));
                    msg_out /= &msg_out
                        .sum_axis(Axis(1))
                        .insert_axis(Axis(1))
                        .broadcast(msg_out.shape())
                        .unwrap();
                    msg_out.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
                },
            );

            distri.mapv_inplace(|x| (2.0 as f64).powf(x));
            distri /= &distri
                .sum_axis(Axis(1))
                .insert_axis(Axis(1))
                .broadcast(distri.shape())
                .unwrap();
            distri.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
        }
    });
}

pub fn update_functions(functions: &PyList, variables: &PyList) {
    let functions: Vec<&PyDict> = functions.iter().map(|x| x.extract().unwrap()).collect();
    functions.iter().for_each(|function| {
        let inputs: Vec<isize> = function.get_item("inputs").unwrap().extract().unwrap();
        let outputs: Vec<isize> = function.get_item("outputs").unwrap().extract().unwrap();
        let func: usize = function.get_item("func").unwrap().extract().unwrap();
        let offset: Vec<isize> = function.get_item("offset").unwrap().extract().unwrap();

        let inputs_v: Vec<&PyDict> = inputs
            .iter()
            .map(|x| variables.get_item(*x).extract().unwrap())
            .collect();

        let outputs_v: Vec<&PyDict> = outputs
            .iter()
            .map(|x| variables.get_item(*x).extract().unwrap())
            .collect();

        // message to send
        let msg: &PyArray3<f64> = function.get_item("msg").unwrap().extract().unwrap();
        let mut msg = unsafe { msg.as_array_mut() };

        // output msg
        let output_msg: &PyArray3<f64> = outputs_v[0].get_item("msg").unwrap().extract().unwrap();
        let mut output_msg = unsafe { output_msg.as_array_mut() };
        let mut output_msg_s = output_msg.slice_mut(s![.., offset[0], ..]);
        let nc = output_msg_s.shape()[1];

        if func == 0 {
            // AND between two distri
            // first input msg
            let input1_msg: &PyArray3<f64> =
                inputs_v[0].get_item("msg").unwrap().extract().unwrap();
            let mut input1_msg = unsafe { input1_msg.as_array_mut() };
            let mut input1_msg_s = input1_msg.slice_mut(s![.., offset[1], ..]);

            // second input msg
            let input2_msg: &PyArray3<f64> =
                inputs_v[1].get_item("msg").unwrap().extract().unwrap();
            let mut input2_msg = unsafe { input2_msg.as_array_mut() };
            let mut input2_msg_s = input2_msg.slice_mut(s![.., offset[2], ..]);
            msg.fill(0.0);
            msg.outer_iter_mut()
                .zip(input1_msg_s.outer_iter_mut())
                .zip(input2_msg_s.outer_iter_mut())
                .zip(output_msg_s.outer_iter_mut())
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
                    },
                );
        } else if func == 1 {
            // XOR between two distri
            if inputs_v.len() == 2 {
                xor_2(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else if inputs_v.len() == 3 {
                xor_3(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else if inputs_v.len() == 4 {
                xor_4(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else if inputs_v.len() == 5 {
                xor_5(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else if inputs_v.len() == 6 {
                xor_6(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else if inputs_v.len() == 7 {
                xor_7(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else if inputs_v.len() == 8 {
                xor_8(inputs_v, offset, &mut output_msg_s, &mut msg, nc);
            } else {
                panic!();
            }
        } else if func == 2 {
            // XOR with array value
            let fixed_inputs: PyReadonlyArray1<u32> =
                inputs_v[1].get_item("values").unwrap().extract().unwrap();
            let fixed_inputs = fixed_inputs.as_array();

            // first input msg
            let input1_msg: &PyArray3<f64> =
                inputs_v[0].get_item("msg").unwrap().extract().unwrap();
            let mut input1_msg = unsafe { input1_msg.as_array_mut() };
            let mut input1_msg_s = input1_msg.slice_mut(s![.., offset[1], ..]);

            msg.outer_iter_mut()
                .zip(fixed_inputs)
                .zip(input1_msg_s.outer_iter())
                .zip(output_msg_s.outer_iter())
                .for_each(|(((mut msg, fixed_input), input_msg), output_msg)| {
                    msg.fill(0.0);
                    let mut msg = msg.slice_mut(s![0..2, ..]);
                    let msg_s = msg.as_slice_mut().unwrap();

                    let input_msg = input_msg.as_slice().unwrap();
                    let output_msg = output_msg.as_slice().unwrap();
                    for i in 0..nc {
                        let o = i as u32 ^ *fixed_input;
                        // message to the output
                        msg_s[o as usize] += input_msg[i as usize];
                        // message to the input
                        msg_s[nc + i as usize] += output_msg[o as usize];
                    }

                    let mut tmp_s = msg.slice_mut(s![0, ..]);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);

                    let mut tmp_s = msg.slice_mut(s![1, ..]);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                });
        } else if func == 3 {
            // XOR with array value
            let table: PyReadonlyArray1<u32> =
                inputs_v[1].get_item("table").unwrap().extract().unwrap();
            let table = table.as_array();
            let table = table.as_slice().unwrap();

            // first input msg
            let input1_msg: &PyArray3<f64> =
                inputs_v[0].get_item("msg").unwrap().extract().unwrap();
            let mut input1_msg = unsafe { input1_msg.as_array_mut() };
            let mut input1_msg_s = input1_msg.slice_mut(s![.., offset[1], ..]);

            msg.outer_iter_mut()
                .zip(input1_msg_s.outer_iter())
                .zip(output_msg_s.outer_iter())
                .for_each(|((mut msg, input_msg), output_msg)| {
                    msg.fill(0.0);
                    let mut msg = msg.slice_mut(s![0..2, ..]);
                    let msg_s = msg.as_slice_mut().unwrap();

                    let input_msg = input_msg.as_slice().unwrap();
                    let output_msg = output_msg.as_slice().unwrap();
                    for i in 0..nc {
                        let o = table[i as usize];
                        // message to the output
                        msg_s[o as usize] += input_msg[i as usize];
                        // message to the input
                        msg_s[nc + i as usize] += output_msg[o as usize];
                    }

                    let mut tmp_s = msg.slice_mut(s![0, ..]);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);

                    let mut tmp_s = msg.slice_mut(s![1, ..]);
                    let s = tmp_s.iter().fold(0.0, |acc, x| acc + *x);
                    tmp_s.iter_mut().for_each(|x| *x /= s);
                });
        } else {
            panic!();
        }
        msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
    });
}

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
}

fn xor_2(
    inputs_v: Vec<&PyDict>,
    offset: Vec<isize>,
    output_msg: &mut ArrayViewMut2<f64>,
    msg: &mut ArrayViewMut3<f64>,
    nc: usize,
) {
    // first input msg
    let input1_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
    let mut input1_msg = unsafe { input1_msg.as_array_mut() };
    let mut input1_msg_s = input1_msg.slice_mut(s![.., offset[1], ..]);

    // first input msg
    let input_msg: &PyArray3<f64> = inputs_v[1].get_item("msg").unwrap().extract().unwrap();
    let mut input_msg = unsafe { input_msg.as_array_mut() };
    let mut input2_msg_s = input_msg.slice_mut(s![.., offset[2], ..]);

    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
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

    let mut tmp_all = Array1::<f64>::ones(nc);
    let tmp_all_s = tmp_all.as_slice_mut().unwrap();
    msg.outer_iter_mut()
        .zip(input1_msg_s.outer_iter_mut())
        .zip(input2_msg_s.outer_iter_mut())
        .zip(input3_msg_s.outer_iter_mut())
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
}
