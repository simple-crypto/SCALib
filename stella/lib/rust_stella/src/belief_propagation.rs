//use ndarray::parallel::prelude::*;
use ndarray::{s, Axis};
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
                },
            );

            let max = msg
                .fold_axis(Axis(2), f64::MIN, |acc, x| acc.max(*x))
                .insert_axis(Axis(2));
            msg -= &max;

            msg.mapv_inplace(|x| (2.0 as f64).powf(x));
            msg /= &msg
                .sum_axis(Axis(2))
                .insert_axis(Axis(2))
                .broadcast(msg.shape())
                .unwrap();
            msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });

            let max = distri
                .fold_axis(Axis(1), f64::MIN, |acc, x| acc.max(*x))
                .insert_axis(Axis(1));
            distri -= &max;
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

        // first input msg
        let input1_msg: &PyArray3<f64> = inputs_v[0].get_item("msg").unwrap().extract().unwrap();
        let mut input1_msg = unsafe { input1_msg.as_array_mut() };
        let mut input1_msg_s = input1_msg.slice_mut(s![.., offset[0], ..]);

        // output msg
        let output_msg: &PyArray3<f64> = outputs_v[0].get_item("msg").unwrap().extract().unwrap();
        let mut output_msg = unsafe { output_msg.as_array_mut() };
        let mut output_msg_s = output_msg.slice_mut(s![.., offset[2], ..]);
        let nc = output_msg_s.shape()[1];

        if func == 1 {
            // XOR between two distri
            // second input msg
            let input2_msg: &PyArray3<f64> =
                inputs_v[1].get_item("msg").unwrap().extract().unwrap();
            let mut input2_msg = unsafe { input2_msg.as_array_mut() };
            let mut input2_msg_s = input2_msg.slice_mut(s![.., offset[1], ..]);
            msg.outer_iter_mut()
                .zip(input1_msg_s.outer_iter_mut())
                .zip(input2_msg_s.outer_iter_mut())
                .zip(output_msg_s.outer_iter_mut())
                .for_each(
                    |(((mut msg, mut input1_msg), mut input2_msg), mut output_msg)| {
                        let input1_msg_s = input1_msg.as_slice_mut().unwrap();
                        let input2_msg_s = input2_msg.as_slice_mut().unwrap();
                        let output_msg_s = output_msg.as_slice_mut().unwrap();
                        fwht(input1_msg_s, nc);
                        fwht(input2_msg_s, nc);
                        fwht(output_msg_s, nc);

                        // message to the output
                        let mut tmp = msg.slice_mut(s![2, ..]);
                        tmp.assign(&(&input2_msg * &input1_msg));
                        let tmp_s = tmp.as_slice_mut().unwrap();
                        fwht(tmp_s, nc);

                        // message to the input 1
                        let mut tmp = msg.slice_mut(s![0, ..]);
                        tmp.assign(&(&input2_msg * &output_msg));
                        let tmp_s = tmp.as_slice_mut().unwrap();
                        fwht(tmp_s, nc);

                        // message to the input 2
                        let mut tmp = msg.slice_mut(s![1, ..]);
                        tmp.assign(&(&input1_msg * &output_msg));
                        let tmp_s = tmp.as_slice_mut().unwrap();
                        fwht(tmp_s, nc);
                    },
                );
        } else if func == 2 {
            // XOR with array value
            let fixed_inputs: PyReadonlyArray1<u32> =
                inputs_v[1].get_item("values").unwrap().extract().unwrap();
            let fixed_inputs = fixed_inputs.as_array();

            msg.fill(0.0);
            msg.outer_iter_mut()
                .zip(fixed_inputs.iter())
                .zip(input1_msg_s.outer_iter())
                .zip(output_msg_s.outer_iter())
                .for_each(|(((mut msg, fixed_input), input_msg), output_msg)| {
                    let mut msg = msg.slice_mut(s![0..3, ..]);
                    let msg_s = msg.as_slice_mut().unwrap();

                    let input_msg = input_msg.as_slice().unwrap();
                    let output_msg = output_msg.as_slice().unwrap();
                    for i in 0..nc {
                        let o = i as u32 ^ *fixed_input;
                        // message to the output
                        msg_s[2 * nc + o as usize] += input_msg[i as usize];
                        // message to the input
                        msg_s[i as usize] += output_msg[o as usize];
                    }
                });
        } else {
            panic!();
        }
        for mut msg in msg.genrows_mut() {
            msg /= msg.sum();
        }
        msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
    });
}
