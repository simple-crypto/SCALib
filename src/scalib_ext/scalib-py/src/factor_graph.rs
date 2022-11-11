
//! Python binding of SCALib's FactorGraph rust implementation.

use std::sync::Arc;
use std::collections::HashMap;

use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::exceptions::{PyTypeError, PyKeyError, PyValueError};
use numpy::{PyArray, PyArray1, PyArray2};

use scalib::sasca;

#[pyclass(module = "_scalib_ext")]
pub(crate) struct FactorGraph {
    inner: Option<Arc<sasca::FactorGraph>>,
}
impl FactorGraph {
    fn get_inner(&self) -> &Arc<sasca::FactorGraph> {
        self.inner.as_ref().unwrap()
    }
    fn get_var(&self, var: &str) -> PyResult<sasca::VarId> {
        self.get_inner().get_varid(var).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_factor(&self, factor: &str) -> PyResult<sasca::FactorId> {
        self.get_inner().get_factorid(factor).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// TODO run stuff on SCALib thread pool

#[pymethods]
impl FactorGraph {
    #[new]
    #[args(args = "*")]
    fn new(args: &PyTuple) -> PyResult<Self> {
        if args.len() == 0 {
            Ok(Self { inner: None })
        } else {
            let (description, tables): (&str, std::collections::HashMap<String, &PyArray1<sasca::ClassVal>>) =
                                         args.extract()?;
            let tables = tables.into_iter().map(|(k, v)| PyResult::<_>::Ok((k, PyArray::to_vec(v)?))).collect::<Result<std::collections::HashMap<_, _>, _>>()?;
            let fg = sasca::build_graph(description, tables).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner: Some(Arc::new(fg)) })
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner.as_deref()).unwrap()).to_object(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = Some(Arc::new(deserialize(s.as_bytes()).unwrap()));
                Ok(())
            }
            Err(e) => Err(e),
        }
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

    pub fn var_names(&self) -> Vec<&str> {
        self.get_inner().var_names().collect()
    }
    pub fn factor_names(&self) -> Vec<&str> {
        self.get_inner().factor_names().collect()
    }
    pub fn factor_scope<'s>(&'s self, factor: &str) -> PyResult<Vec<&'s str>> {
        let factor_id = self.get_factor(factor)?;
        Ok(self.get_inner().factor_scope(factor_id).map(|v| self.get_inner().var_name(v)).collect())
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
    fn get_factor(&self, factor: &str) -> PyResult<sasca::FactorId> {
        self.get_inner().get_graph().get_factorid(factor).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_edge(&self, var: sasca::VarId, factor: sasca::FactorId) -> PyResult<sasca::EdgeId> {
        self.get_inner().get_graph().edge(var, factor).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_edge_named(&self, var: &str, factor: &str) -> PyResult<sasca::EdgeId> {
        self.get_edge(self.get_var(var)?, self.get_factor(factor)?)
    }
}

#[pymethods]
impl BPState {
    #[new]
    #[args(_args = "*")]
    fn new(_args: &PyTuple) -> PyResult<Self> {
        Ok(Self { inner: None })
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
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
        let var_id = self.get_var(var)?;
        self.get_inner_mut().drop_evidence(var_id);
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
    pub fn drop_state(&mut self, var: &str) -> PyResult<()> {
        let var_id = self.get_var(var)?;
        self.get_inner_mut().drop_state(var_id);
        Ok(())
    }
    pub fn get_belief_to_var(&self, py: Python, var: &str, factor: &str) -> PyResult<PyObject> {
        let edge_id = self.get_edge_named(var, factor)?;
        distr2py(py, self.get_inner().get_belief_to_var(edge_id))
    }
    pub fn get_belief_from_var(&self, py: Python, var: &str, factor: &str) -> PyResult<PyObject> {
        let edge_id = self.get_edge_named(var, factor)?;
        distr2py(py, self.get_inner().get_belief_from_var(edge_id))
    }
    pub fn propagate_var(&mut self, var: &str) -> PyResult<()> {
        let var_id = self.get_var(var)?;
        self.get_inner_mut().propagate_var(var_id);
        Ok(())
    }
    pub fn propagate_factor_all(&mut self, factor: &str) -> PyResult<()> {
        let factor_id = self.get_factor(factor)?;
        self.get_inner_mut().propagate_factor_all(factor_id);
        Ok(())
    }
    pub fn propagate_factor(&mut self, factor: &str, dest: Vec<&str>) -> PyResult<()> {
        let factor_id = self.get_factor(factor)?;
        let dest = dest.iter().map(|v| self.get_var(v)).collect::<Result<Vec<_>, _>>()?;
        self.get_inner_mut().propagate_factor(factor_id, dest.as_slice());
        Ok(())
    }
    pub fn propagate_loopy_step(&mut self, n_steps: u32) {
        self.get_inner_mut().propagate_loopy_step(n_steps);
    }
    pub fn graph(&self) -> FactorGraph {
        FactorGraph { inner: Some(self.get_inner().get_graph().clone()) }
    }
}


fn obj2distr(py: Python, distr: PyObject, multi: bool) -> PyResult<sasca::Distribution> {
    Ok(if multi {
        let distr: &PyArray2<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_multi(distr.to_owned_array())
    } else {
        let distr: &PyArray1<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_single(distr.to_owned_array())
    })
}

fn distr2py(py: Python, distr: &sasca::Distribution) -> PyResult<PyObject> {
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
