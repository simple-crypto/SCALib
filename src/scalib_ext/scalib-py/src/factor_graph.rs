//! Python binding of SCALib's FactorGraph rust implementation.

use std::collections::HashMap;
use std::sync::Arc;

use bincode::{deserialize, serialize};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyList, PyTuple};

use scalib::sasca;

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct FactorGraph {
    inner: Option<Arc<sasca::FactorGraph>>,
}
impl FactorGraph {
    fn get_inner(&self) -> &Arc<sasca::FactorGraph> {
        self.inner.as_ref().unwrap()
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
    fn new(args: &Bound<PyTuple>) -> PyResult<Self> {
        if args.len() == 0 {
            Ok(Self { inner: None })
        } else {
            let (description, tables): (
                PyBackedStr,
                std::collections::HashMap<String, PyReadonlyArray1<sasca::ClassVal>>,
            ) = args.extract()?;
            let tables = tables
                .into_iter()
                .map(|(k, v)| PyResult::<_>::Ok((k, v.as_slice()?.to_vec())))
                .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
            let fg = sasca::build_graph(description.as_ref(), tables)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Some(Arc::new(fg)),
            })
        }
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let to_ser: Option<&sasca::FactorGraph> = self.inner.as_deref();
        PyBytes::new(py, &serialize(&to_ser).unwrap())
    }

    pub fn __setstate__(&mut self, state: &Bound<PyBytes>) {
        let deser: Option<sasca::FactorGraph> = deserialize(state.as_bytes()).unwrap();
        self.inner = deser.map(Arc::new);
    }

    pub fn new_bp(
        &self,
        py: Python,
        nmulti: u32,
        public_values: PyObject,
        gen_factors: PyObject,
    ) -> PyResult<BPState> {
        let pub_values = pyobj2pubs(
            py,
            public_values,
            self.get_inner().public_multi(),
            self.get_inner().nc(),
        )?;
        let gen_factors = pyobj2factors(py, gen_factors, self.get_inner().gf_multi())?;
        Ok(BPState {
            inner: Some(sasca::BPState::new(
                self.get_inner().clone(),
                nmulti,
                pub_values,
                gen_factors,
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
        factor_assignments: PyObject,
    ) -> PyResult<()> {
        let inner = self.get_inner();
        let pub_values = pyobj2pubs(py, public_values, inner.public_multi(), inner.nc())?;
        let var_values = pyobj2pubs(
            py,
            var_assignments,
            inner.vars().map(|(v, vn)| (vn, inner.var_multi(v))),
            inner.nc(),
        )?;
        let gen_factors = pyobj2factors(py, factor_assignments, self.get_inner().gf_multi())?;
        inner
            .sanity_check(pub_values, var_values.into(), gen_factors)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

fn pyobj2pubs<'a>(
    py: Python,
    public_values: PyObject,
    expected: impl Iterator<Item = (&'a str, bool)>,
    nc: usize,
) -> PyResult<Vec<sasca::PublicValue>> {
    // FIXME: move back to &str instead of String
    let mut public_values: HashMap<String, PyObject> = public_values.extract(py)?;
    let pubs = expected
        .map(|(pub_name, multi)| {
            obj2pub(
                py,
                public_values
                    .remove(pub_name)
                    .ok_or_else(|| PyKeyError::new_err(format!("Missing value {}.", pub_name)))?,
                multi,
                nc,
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

fn pyobj2genfactor_inner(obj: &Bound<PyAny>) -> PyResult<sasca::GenFactorInner> {
    let kind: u32 = obj.getattr("kind")?.extract()?;
    let dense: u32 = obj.getattr("GenFactorKind")?.getattr("DENSE")?.extract()?;
    let sparse_functional: u32 = obj
        .getattr("GenFactorKind")?
        .getattr("SPARSE_FUNCTIONAL")?
        .extract()?;
    if kind == dense {
        let factor: numpy::PyReadonlyArrayDyn<f64> = obj.getattr("factor")?.extract()?;
        let factor = factor.as_array().as_standard_layout().into_owned();
        Ok(sasca::GenFactorInner::Dense(factor))
    } else if kind == sparse_functional {
        let factor: numpy::PyReadonlyArray2<sasca::ClassVal> = obj.getattr("factor")?.extract()?;
        let factor = factor.as_array().as_standard_layout().into_owned();
        Ok(sasca::GenFactorInner::SparseFunctional(factor))
    } else {
        Err(PyValueError::new_err((
            "Unknown kind",
            obj.getattr("kind")?.unbind(),
        )))
    }
}

fn pyobj2factors<'a>(
    py: Python,
    gen_factors: PyObject,
    expected: impl Iterator<Item = (&'a str, bool)>,
) -> PyResult<Vec<sasca::GenFactor>> {
    // TODO validate single/para, dimensionality and dimensions.
    // FIXME: move back to &str instead of String
    let mut gen_factors: HashMap<String, PyObject> = gen_factors.extract(py)?;
    let res = expected
        .map(|(name, multi)| {
            let gf = gen_factors
                .remove(name)
                .ok_or_else(|| PyKeyError::new_err(format!("Missing gen factor {}.", name)))?;
            if multi {
                if gf.downcast_bound::<PyList>(py).is_err() {
                    return Err(PyTypeError::new_err(format!(
                        "Generalized factor {} must be a list, as it is MULTI.",
                        name
                    )));
                }
                let obj: Vec<Bound<PyAny>> = gf.extract(py)?;
                Ok(sasca::GenFactor::Multi(
                    obj.into_iter()
                        .map(|obj| pyobj2genfactor_inner(&obj))
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            } else {
                let obj: Bound<PyAny> = gf.extract(py)?;
                Ok(sasca::GenFactor::Single(pyobj2genfactor_inner(&obj)?))
            }
        })
        .collect::<Result<Vec<sasca::GenFactor>, PyErr>>()?;
    if gen_factors.is_empty() {
        Ok(res)
    } else {
        let unknown = gen_factors.keys().collect::<Vec<_>>();
        Err(PyKeyError::new_err(if unknown.len() == 1 {
            format!("{} is not a generalized factor.", unknown[0])
        } else {
            format!("{:?} are not generalized factors.", unknown)
        }))
    }
}

#[pyclass(module = "scalib._scalib_ext")]
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
    fn new(_args: &Bound<PyTuple>) -> PyResult<Self> {
        Ok(Self { inner: None })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &serialize(&self.inner).unwrap())
    }

    pub fn __setstate__(&mut self, state: &Bound<PyBytes>) {
        self.inner = deserialize(state.as_bytes()).unwrap();
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
    pub fn get_state<'py>(&self, py: Python<'py>, var: &str) -> PyResult<Bound<'py, PyAny>> {
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
    pub fn get_belief_to_var<'py>(
        &self,
        py: Python<'py>,
        var: &str,
        factor: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let edge_id = self.get_edge_named(var, factor)?;
        distr2py(py, self.get_inner().get_belief_to_var(edge_id))
    }
    pub fn get_belief_from_var<'py>(
        &self,
        py: Python<'py>,
        var: &str,
        factor: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let edge_id = self.get_edge_named(var, factor)?;
        distr2py(py, self.get_inner().get_belief_from_var(edge_id))
    }
    pub fn propagate_var(
        &mut self,
        py: Python,
        var: &str,
        config: crate::ConfigWrapper,
        clear_beliefs: bool,
    ) -> PyResult<()> {
        config.on_worker(py, |_| {
            let var_id = self.get_var(var)?;
            self.get_inner_mut().propagate_var(var_id, clear_beliefs);
            Ok(())
        })
    }
    pub fn propagate_var_to(
        &mut self,
        py: Python,
        var: &str,
        dest: Vec<PyBackedStr>,
        config: crate::ConfigWrapper,
        clear_beliefs: bool,
        clear_evidence: bool,
    ) -> PyResult<()> {
        config.on_worker(py, |_| {
            let var_id = self.get_var(var)?;
            let edge_ids = dest
                .iter()
                .map(|d| self.get_edge(var_id, self.get_factor(d)?))
                .collect::<Result<Vec<_>, _>>()?;
            self.get_inner_mut()
                .propagate_var_to(var_id, edge_ids, clear_beliefs, clear_evidence);
            Ok(())
        })
    }
    pub fn propagate_all_vars(
        &mut self,
        py: Python,
        config: crate::ConfigWrapper,
        clear_beliefs: bool,
    ) -> PyResult<()> {
        config.on_worker(py, |_| {
            self.get_inner_mut().propagate_all_vars(clear_beliefs);
            Ok(())
        })
    }
    pub fn propagate_factor_all(
        &mut self,
        py: Python,
        factor: &str,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        config.on_worker(py, |_| {
            let factor_id = self.get_factor(factor)?;
            self.get_inner_mut().propagate_factor_all(factor_id);
            Ok(())
        })
    }
    pub fn set_belief_from_var(
        &mut self,
        py: Python,
        var: &str,
        factor: &str,
        distr: PyObject,
    ) -> PyResult<()> {
        let edge_id = self.get_edge_named(var, factor)?;
        let bp = self.get_inner_mut();
        let distr = obj2distr(py, distr, bp.get_graph().edge_multi(edge_id))?;
        bp.set_belief_from_var(edge_id, distr)
            .map_err(|e| PyTypeError::new_err(e.to_string()))?;
        Ok(())
    }
    pub fn set_belief_to_var(
        &mut self,
        py: Python,
        var: &str,
        factor: &str,
        distr: PyObject,
    ) -> PyResult<()> {
        let edge_id = self.get_edge_named(var, factor)?;
        let bp = self.get_inner_mut();
        let distr = obj2distr(py, distr, bp.get_graph().edge_multi(edge_id))?;
        bp.set_belief_to_var(edge_id, distr)
            .map_err(|e| PyTypeError::new_err(e.to_string()))?;
        Ok(())
    }
    pub fn propagate_factor(
        &mut self,
        py: Python,
        factor: &str,
        dest: Vec<PyBackedStr>,
        clear_incoming: bool,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        config.on_worker(py, |_| {
            let factor_id = self.get_factor(factor)?;
            let dest = dest
                .iter()
                .map(|v| self.get_var(v.as_ref()))
                .collect::<Result<Vec<_>, _>>()?;
            self.get_inner_mut()
                .propagate_factor(factor_id, dest.as_slice(), clear_incoming);
            Ok(())
        })
    }
    pub fn propagate_loopy_step(
        &mut self,
        py: Python,
        n_steps: u32,
        config: crate::ConfigWrapper,
        clear_beliefs: bool,
    ) {
        config.on_worker(py, |_| {
            self.get_inner_mut()
                .propagate_loopy_step(n_steps, clear_beliefs);
        });
    }
    pub fn graph(&self) -> FactorGraph {
        FactorGraph {
            inner: Some(self.get_inner().get_graph().clone()),
        }
    }
    pub fn propagate_acyclic(
        &mut self,
        py: Python,
        dest: &str,
        clear_intermediates: bool,
        clear_evidence: bool,
        config: crate::ConfigWrapper,
    ) -> PyResult<()> {
        config.on_worker(py, |_| {
            let var = self.get_var(dest)?;
            self.get_inner_mut()
                .propagate_acyclic(var, clear_intermediates, clear_evidence)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
}

fn obj2distr(py: Python, distr: PyObject, multi: bool) -> PyResult<sasca::Distribution> {
    if multi {
        let distr: PyReadonlyArray2<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_multi(distr.as_array().as_standard_layout().into_owned())
    } else {
        let distr: PyReadonlyArray1<f64> = distr.extract(py)?;
        sasca::Distribution::from_array_single(distr.as_array().as_standard_layout().into_owned())
    }
    .map_err(|e| PyTypeError::new_err(e.to_string()))
}

fn obj2pub(py: Python, obj: PyObject, multi: bool, nc: usize) -> PyResult<sasca::PublicValue> {
    Ok(if multi {
        let obj: Vec<sasca::ClassVal> = obj.extract(py)?;
        if let Some(v) = obj.iter().filter(|x| **x as usize >= nc).next() {
            return Err(PyValueError::new_err(format!(
                "Public value larger than NC ({} >= {}).",
                *v, nc
            )));
        }
        sasca::PublicValue::Multi(obj)
    } else {
        let obj: sasca::ClassVal = obj.extract(py)?;
        sasca::PublicValue::Single(obj)
    })
}

fn distr2py<'py>(py: Python<'py>, distr: &sasca::Distribution) -> PyResult<Bound<'py, PyAny>> {
    if let Some(d) = distr.value() {
        if distr.multi() {
            return Ok(PyArray2::from_array(py, &d).into_any());
        } else {
            return Ok(PyArray1::from_array(py, &d.slice(ndarray::s![0, ..])).into_any());
        }
    } else {
        return Ok(py.None().into_bound(py));
    }
}
