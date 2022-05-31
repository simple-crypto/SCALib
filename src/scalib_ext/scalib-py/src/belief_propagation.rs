//! Python binding of SCALib's belief propagation.

use crate::thread_pool::ThreadPool;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;

use scalib::belief_propagation::{Func, FuncType, Var, VarType};

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
    } else if func == 3 {
        let table: PyReadonlyArray1<u32> = function.get_item("table").unwrap().extract().unwrap();
        f = FuncType::LOOKUP(table.as_array().to_owned());
    } else if func == 4 {
        let values: PyReadonlyArray1<u32> = function.get_item("values").unwrap().extract().unwrap();
        f = FuncType::ANDCST(values.as_array().to_owned());
    } else if func == 5 {
        f = FuncType::ADD;
    } else if func == 6 {
        let values: PyReadonlyArray1<u32> = function.get_item("values").unwrap().extract().unwrap();
        f = FuncType::ADDCST(values.as_array().to_owned());
    } else if func == 7 {
        f = FuncType::MUL;
    } else if func == 8 {
        let values: PyReadonlyArray1<u32> = function.get_item("values").unwrap().extract().unwrap();
        f = FuncType::MULCST(values.as_array().to_owned());
    } else {
        panic!("func {} value is not recognized", func);
    }

    Func {
        neighboors: neighboors,
        functype: f,
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
    vertex: usize,
    // size of the field
    nc: usize,
    // number of copies in the graph (n_runs)
    n: usize,
    // show a progress bar
    progress: bool,
    thread_pool: &ThreadPool,
) -> PyResult<()> {
    // map all python functions to rust ones + generate the mapping in vec_functs_id
    let functions_rust: Vec<Func> = functions
        .iter()
        .map(|x| to_func(x.downcast::<PyDict>().unwrap()))
        .collect();

    // map all python var to rust ones
    // generate the edge mapping in vec_vars_id
    // init the messages along the edges with initial distributions
    let mut variables_rust: Vec<Var> = variables
        .iter()
        .map(|x| to_var(x.downcast::<PyDict>().unwrap()))
        .collect();

    crate::on_worker(py, thread_pool, || {
        scalib::belief_propagation::run_bp(
            &functions_rust,
            &mut variables_rust,
            it,
            vertex,
            nc,
            n,
            progress,
        )
        .unwrap();
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
