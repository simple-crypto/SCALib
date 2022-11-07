
//! Python binding of SCALib's FactorGraph rust implementation.

use std::rc::Rc;

//use bincode::{deserialize, serialize};
use numpy::{PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

#[pyclass(module = "_scalib_ext")]
pub(crate) struct FactorGraph {
    inner: Option<Rc<scalib::sasca::FactorGraph>>,
}


#[pymethods]
impl FactorGraph {
    #[new]
    #[args(args = "*")]
    fn new(py: Python, args: &PyTuple) -> PyResult<Self> {
        if args.len() == 0 {
            Ok(Self { inner: None })
        } else {
            /*
            let (a, b): ( PyReadonlyArray2<f64>, bool) = args.extract()?;
            let a = a.as_array().to_owned();
            let inner = py.allow_threads(|| {
                Some(...)
            });
            Ok(Self { inner })
            */
            todo!("FactorGraph deserialization");
        }
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                todo!("Implement Deserialize for FactorGraph");
                //self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    #[staticmethod]
    pub fn from_desc(py: Python, description: &str, tables: std::collections::HashMap<String, &PyArray1<scalib::sasca::ClassVal>>) -> PyResult<Self> {
        let tables = tables.into_iter().map(|(k, v)| PyResult::<_>::Ok((k, PyArray::to_vec(v)?))).collect::<Result<std::collections::HashMap<_, _>, _>>()?;
        let fg = scalib::sasca::build_graph(description, tables)?;
        Ok(Self { inner: Some(Rc::new(fg)) })
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        todo!("Implement Serialize for FactorGraph")
        //Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }
}


