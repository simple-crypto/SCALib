//use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Axis};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::types::{PyDict, PyList};

pub fn update_variables(functions: &PyList, variables: &PyList) {
    variables.iter().for_each(|var| {
        let var = var.downcast::<PyDict>().unwrap();
        if var.contains("distri").unwrap() {
            let distri: &PyArray2<f64> = var.get_item("distri").unwrap().extract().unwrap();
            let mut distri = unsafe { distri.as_array_mut() };
            let msg: &PyArray3<f64> = var.get_item("msg").unwrap().extract().unwrap();
            let mut msg = unsafe { msg.as_array_mut() };

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
                println!("distri_orig {:?}", distri_orig);
                distri.assign(&distri_orig);
                distri.mapv_inplace(|x| f64::log2(x));
            } else {
                distri.fill(0.0);
            }
            println!("distri_log {:?}", distri);

            offset
                .iter()
                .zip(neighboors.iter())
                .for_each(|(offset, neighboor)| {
                    let msg_in: &PyArray3<f64> =
                        neighboor.get_item("msg").unwrap().extract().unwrap();
                    let mut msg_in = unsafe { msg_in.as_array_mut() };
                    let mut msg_in = msg_in.slice_mut(s![.., *offset, ..]);
                    msg_in.mapv_inplace(|x| f64::log2(x));
                    distri += &msg_in;
                });
            offset.iter().zip(neighboors.iter()).enumerate().for_each(
                |(i, (offset, neighboor))| {
                    let msg_in: &PyArray3<f64> =
                        neighboor.get_item("msg").unwrap().extract().unwrap();
                    let mut msg_in = unsafe { msg_in.as_array_mut() };
                    let mut msg_in = msg_in.slice_mut(s![.., *offset, ..]);
                    let mut msg_out = msg.slice_mut(s![.., i, ..]);
                    msg_out.assign(&(&distri - &msg_in));
                    println!("msg_out {:?}", msg_out);
                },
            );

            for mut row in msg.genrows_mut() {
                row.mapv_inplace(|x| (2.0 as f64).powf(x));
                row /= row.sum();
                row.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
            }
            for mut row in distri.genrows_mut() {
                println!("row distri log {:?}",row);
                row.mapv_inplace(|x| (2.0 as f64).powf(x));
                println!("row distri {:?}",row);
                row /= row.sum();
                row.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
            }
        }
    });
}

pub fn update_functions(functions: &PyList, variables: &PyList) {
    functions.iter().enumerate().for_each(|(it, function)| {
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
        let msg: &PyArray3<f64> = function.get_item("msg").unwrap().extract().unwrap();
        let mut msg = unsafe { msg.as_array_mut() };

        if func == 2 {
            let fixed_inputs: PyReadonlyArray1<u32> =
                inputs_v[1].get_item("values").unwrap().extract().unwrap();
            let fixed_inputs = fixed_inputs.as_array();

            let input_msg: PyReadonlyArray3<f64> =
                inputs_v[0].get_item("msg").unwrap().extract().unwrap();
            let input_msg = input_msg.as_array();
            let input_msg_s = input_msg.slice(s![.., offset[0], ..]);

            let output_msg: PyReadonlyArray3<f64> =
                outputs_v[0].get_item("msg").unwrap().extract().unwrap();
            let output_msg = output_msg.as_array();
            let output_msg_s = output_msg.slice(s![.., offset[2], ..]);
            let nc = output_msg_s.shape()[1];

            msg.fill(0.0);
            msg.outer_iter_mut()
                .zip(fixed_inputs.iter())
                .zip(input_msg_s.outer_iter())
                .zip(output_msg_s.outer_iter())
                .for_each(|(((mut msg, fixed_input), input_msg), output_msg)| {
                    let mut msg = msg.slice_mut(s![0..3, ..]);
                    let msg_s = msg.as_slice_mut().unwrap();

                    let input_msg = input_msg.as_slice().unwrap();
                    let output_msg = output_msg.as_slice().unwrap();
                    for i in 0..nc {
                        let o = i as u32 ^ *fixed_input;
                        msg_s[2 * nc + i as usize] += input_msg[o as usize];
                        msg_s[o as usize] += output_msg[i as usize];
                    }
                });
            for mut msg in msg.genrows_mut() {
                msg /= msg.sum();
            }
            msg.mapv_inplace(|x| if x < 1E-50 { 1E-50 } else { x });
        } else {
            panic!();
        }
    });
}
