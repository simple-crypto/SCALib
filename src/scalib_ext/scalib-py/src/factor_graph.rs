
//! Python binding of SCALib's FactorGraph rust implementation.

use std::sync::Arc;
use std::collections::HashMap;

//use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::exceptions::{PyTypeError, PyKeyError, PyValueError};
use numpy::{PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

use scalib::sasca;

#[pyclass(module = "_scalib_ext")]
pub(crate) struct FactorGraph {
    inner: Option<Arc<sasca::FactorGraph>>,
}
impl FactorGraph {
    fn get_inner(&self) -> &Arc<sasca::FactorGraph> {
        self.inner.as_ref().unwrap()
    }
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

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        todo!("Implement Serialize for FactorGraph")
        //Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    #[staticmethod]
    pub fn from_desc(description: &str, tables: std::collections::HashMap<String, &PyArray1<sasca::ClassVal>>) -> PyResult<Self> {
        let tables = tables.into_iter().map(|(k, v)| PyResult::<_>::Ok((k, PyArray::to_vec(v)?))).collect::<Result<std::collections::HashMap<_, _>, _>>()?;
        let fg = sasca::build_graph(description, tables).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: Some(Arc::new(fg)) })
    }

    pub fn new_bp(&self, py: Python, nmulti: u32, public_values: PyObject) -> PyResult<BPState> {
        let mut public_values: HashMap<&str, ClassValOrList> = public_values.extract(py)?;
        let pub_values = self.get_inner().public_multi().map(|(pub_name, multi)| {
            let v = public_values.remove(pub_name).ok_or_else(|| PyKeyError::new_err(format!("Missing public value {}.", pub_name)))?;
            match (v, multi) {
                (ClassValOrList::Single(v), false) => Ok(sasca::PublicValue::Single(v)),
                (ClassValOrList::List(v), true) => Ok(sasca::PublicValue::Multi(v)),
                (ClassValOrList::Single(_), true) => Err(PyTypeError::new_err(format!("Public value {} is multi, found single value.", pub_name))),
                (ClassValOrList::List(_), false) => Err(PyTypeError::new_err(format!("Public value {} is not multi, found list of values.", pub_name))),
            }
        }).collect::<Result<Vec<sasca::PublicValue>, PyErr>>()?;
        Ok(BPState { inner: Some(sasca::BPState::new(self.get_inner().clone(), nmulti, pub_values)) })
    }
}

#[derive(Debug, Clone, FromPyObject)]
enum ClassValOrList {
    Single(sasca::ClassVal),
    List(Vec<sasca::ClassVal>),
}



#[pyclass(module = "_scalib_ext")]
pub(crate) struct BPState {
    inner: Option<sasca::BPState>,
}
impl BPState {
    fn get_inner(&self) -> &sasca::BPState {
        self.inner.as_ref().unwrap()
    }
    fn get_inner_mut(&mut self) -> &mut sasca::BPState {
        self.inner.as_mut().unwrap()
    }
    fn get_var(&self, var: &str) -> PyResult<sasca::VarId> {
        self.get_inner().get_graph().get_varid(var).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl BPState {
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
            todo!("BPState deserialization");
        }
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                todo!("Implement Deserialize for BPState");
                //self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        todo!("Implement Serialize for BPState")
        //Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    pub fn is_cyclic(&self) -> bool {
        self.get_inner().is_cyclic()
    }

    pub fn set_evidence(&mut self, py: Python, var: &str, distr: PyObject) -> PyResult<()> {
        let var_id = self.get_var(var)?;
        let bp = self.get_inner_mut();
        let distr = obj2distr(py, distr, bp.get_graph().var_multi(var_id))?;
        bp.set_evidence(var_id, distr).map_err(|e| PyTypeError::new_err(e.to_string()))?;
        Ok(())
    }
    pub fn drop_evidence(&mut self, var: &str) -> PyResult<()> {
        self.get_inner_mut().drop_evidence(self.get_var(var)?);
        Ok(())
    }
    pub fn get_state(&self, py: Python, var: &str) -> PyResult<PyObject> {
        distr2py(py, self.get_inner().get_state(self.get_var(var)?))
    }
    pub fn set_state(&mut self, py: Python, var: &str, distr: PyObject) -> PyResult<()> {
        let var_id = self.get_var(var)?;
        let bp = self.get_inner_mut();
        let distr = obj2distr(py, distr, bp.get_graph().var_multi(var_id))?;
        bp.set_state(var_id, distr).map_err(|e| PyTypeError::new_err(e.to_string()))?;
        Ok(())
    }
    pub fn get_belief_to_var(&self, py: Python, var: &str) -> PyResult<PyObject> {
        distr2py(py, self.get_inner().get_belief_to_var(self.get_var(var)?))
    }
    pub fn get_belief_from_var(&self, py: Python, var: &str) -> PyResult<PyObject> {
        distr2py(py, self.get_inner().get_belief_from_var(self.get_var(var)?))
    }
    pub fn propagate_to_var(&mut self, var: &str) -> PyResult<()> {
        self.get_inner_mut().propagate_to_var(self.get_var(var)?);
        Ok(())
    }
    // TODO implement factor naming
    pub fn propagate_factor(&mut self, factor_id: usize, dest: &[&str]) -> PyResult<()> {
        let dest = dest.iter().map(|v| self.get_var(v)).collect::<Result<Vec<_>, _>>()?;
        self.get_inner_mut().propagate_factor(factor_id, dest.as_slice());
        Ok(())
    }
}


fn obj2distr(py: Python, distr: PyObject, multi: bool) -> PyResult<sasca::Distribution> {
    Ok(if multi {
        let distr: &PyArray1<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_single(distr.to_owned_array())
    } else {
        let distr: &PyArray2<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_multi(distr.to_owned_array())
    })
}

fn dist2py(py: Python, distr: &sasca::Distribution) -> PyResult<PyObject> {
    if let Some(d) = distr.value() {
        if distr.multi() {
            return Ok(PyArray2::from_array(py, &d).into_py(py));
        } else {
            return Ok(PyArray1::from_array(py, &d.slice(ndarray::s![0,..])).into_py(py));
        }
    } else {
        return Ok(py.None());
    }
}
