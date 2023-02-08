//! Python binding of SCALib's FactorGraph rust implementation.

use std::collections::HashMap;
use std::sync::Arc;

use bincode::{deserialize, serialize};
use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

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
        self.get_inner()
            .get_varid(var)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_factor(&self, factor: &str) -> PyResult<sasca::FactorId> {
        self.get_inner()
            .get_factorid(factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// TODO run stuff on SCALib thread pool

#[pymethods]
impl FactorGraph {
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &PyTuple) -> PyResult<Self> {
        if args.len() == 0 {
            Ok(Self { inner: None })
        } else {
            let (description, tables): (
                &str,
                std::collections::HashMap<String, &PyArray1<sasca::ClassVal>>,
            ) = args.extract()?;
            let tables = tables
                .into_iter()
                .map(|(k, v)| PyResult::<_>::Ok((k, PyArray::to_vec(v)?)))
                .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
            let fg = sasca::build_graph(description, tables)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Some(Arc::new(fg)),
            })
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
        let pub_values = pyobj2pubs(py, public_values, self.get_inner().public_multi())?;
        Ok(BPState {
            inner: Some(sasca::BPState::new(
                self.get_inner().clone(),
                nmulti,
                pub_values,
            )),
        })
    }

    pub fn var_names(&self) -> Vec<&str> {
        self.get_inner().var_names().collect()
    }
    pub fn factor_names(&self) -> Vec<&str> {
        self.get_inner().factor_names().collect()
    }
    pub fn factor_scope<'s>(&'s self, factor: &str) -> PyResult<Vec<&'s str>> {
        let factor_id = self.get_factor(factor)?;
        Ok(self
            .get_inner()
            .factor_scope(factor_id)
            .map(|v| self.get_inner().var_name(v))
            .collect())
    }
    pub fn sanity_check(
        &self,
        py: Python,
        public_values: PyObject,
        var_assignments: PyObject,
    ) -> PyResult<()> {
        let inner = self.get_inner();
        let pub_values = pyobj2pubs(py, public_values, inner.public_multi())?;
        let var_values = pyobj2pubs(
            py,
            var_assignments,
            inner.vars().map(|(v, vn)| (vn, inner.var_multi(v))),
        )?;
        inner
            .sanity_check(pub_values, var_values.into())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

fn pyobj2pubs<'a>(
    py: Python,
    public_values: PyObject,
    expected: impl Iterator<Item = (&'a str, bool)>,
) -> PyResult<Vec<sasca::PublicValue>> {
    let mut public_values: HashMap<&str, PyObject> = public_values.extract(py)?;
    let pubs = expected
        .map(|(pub_name, multi)| {
            obj2pub(
                py,
                public_values
                    .remove(pub_name)
                    .ok_or_else(|| PyKeyError::new_err(format!("Missing value {}.", pub_name)))?,
                multi,
            )
        })
        .collect::<Result<Vec<sasca::PublicValue>, PyErr>>()?;
    if public_values.is_empty() {
        Ok(pubs)
    } else {
        let unknown_pubs = public_values.keys().collect::<Vec<_>>();
        Err(PyKeyError::new_err(if unknown_pubs.len() == 1 {
            format!("{} is not a public.", unknown_pubs[0])
        } else {
            format!("{:?} are not publics.", unknown_pubs)
        }))
    }
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
        self.get_inner()
            .get_graph()
            .get_varid(var)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_factor(&self, factor: &str) -> PyResult<sasca::FactorId> {
        self.get_inner()
            .get_graph()
            .get_factorid(factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_edge(&self, var: sasca::VarId, factor: sasca::FactorId) -> PyResult<sasca::EdgeId> {
        self.get_inner()
            .get_graph()
            .edge(var, factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn get_edge_named(&self, var: &str, factor: &str) -> PyResult<sasca::EdgeId> {
        self.get_edge(self.get_var(var)?, self.get_factor(factor)?)
    }
}

#[pymethods]
impl BPState {
    #[new]
    #[pyo3(signature = (*_args))]
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
        bp.set_evidence(var_id, distr)
            .map_err(|e| PyTypeError::new_err(e.to_string()))?;
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
        bp.set_state(var_id, distr)
            .map_err(|e| PyTypeError::new_err(e.to_string()))?;
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
    pub fn propagate_all_vars(&mut self) -> PyResult<()> {
        self.get_inner_mut().propagate_all_vars();
        Ok(())
    }
    pub fn propagate_factor_all(&mut self, factor: &str) -> PyResult<()> {
        let factor_id = self.get_factor(factor)?;
        self.get_inner_mut().propagate_factor_all(factor_id);
        Ok(())
    }
    pub fn propagate_factor(
        &mut self,
        factor: &str,
        dest: Vec<&str>,
        clear_incoming: bool,
    ) -> PyResult<()> {
        let factor_id = self.get_factor(factor)?;
        let dest = dest
            .iter()
            .map(|v| self.get_var(v))
            .collect::<Result<Vec<_>, _>>()?;
        self.get_inner_mut()
            .propagate_factor(factor_id, dest.as_slice(), clear_incoming);
        Ok(())
    }
    pub fn propagate_loopy_step(&mut self, n_steps: u32) {
        self.get_inner_mut().propagate_loopy_step(n_steps);
    }
    pub fn graph(&self) -> FactorGraph {
        FactorGraph {
            inner: Some(self.get_inner().get_graph().clone()),
        }
    }
    pub fn propagate_acyclic(
        &mut self,
        dest: &str,
        clear_intermediates: bool,
        clear_evidence: bool,
    ) -> PyResult<()> {
        let var = self.get_var(dest)?;
        self.get_inner_mut()
            .propagate_acyclic(var, clear_intermediates, clear_evidence)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

fn obj2distr(py: Python, distr: PyObject, multi: bool) -> PyResult<sasca::Distribution> {
    if multi {
        let distr: &PyArray2<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_multi(
            distr
                .readonly()
                .as_array()
                .as_standard_layout()
                .into_owned(),
        )
    } else {
        let distr: &PyArray1<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_single(
            distr
                .readonly()
                .as_array()
                .as_standard_layout()
                .into_owned(),
        )
    }
    .map_err(|e| PyTypeError::new_err(e.to_string()))
}

fn obj2pub(py: Python, obj: PyObject, multi: bool) -> PyResult<sasca::PublicValue> {
    Ok(if multi {
        let obj: Vec<sasca::ClassVal> = obj.extract(py)?;
        sasca::PublicValue::Multi(obj)
    } else {
        let obj: sasca::ClassVal = obj.extract(py)?;
        sasca::PublicValue::Single(obj)
    })
}

fn distr2py(py: Python, distr: &sasca::Distribution) -> PyResult<PyObject> {
    if let Some(d) = distr.value() {
        if distr.multi() {
            return Ok(PyArray2::from_array(py, &d).into_py(py));
        } else {
            return Ok(PyArray1::from_array(py, &d.slice(ndarray::s![0, ..])).into_py(py));
        }
    } else {
        return Ok(py.None());
    }
}
