use ndarray::{s, Array1, Array2, Array3, Axis, Zip};
use num_integer::binomial;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct Ttest {
    /// raw moment of order 1 with shape (2,ns)
    m: Array2<f64>,
    /// central sum up to orer d*2 with shape (2,2*d,ns)
    cs: Array3<f64>,
    /// number of samples per class (2,)
    n_samples: Array1<u64>,
    /// order of the test
    d: usize,
    /// number of samples in a trace
    ns: usize,
}
#[pymethods]
impl Ttest {
    #[new]
    /// Create a new Ttest state.
    /// ns: traces length
    /// d: order of the Ttest
    fn new(ns: usize, d: usize) -> Self {
        Ttest {
            m: Array2::<f64>::zeros((2, ns)),
            cs: Array3::<f64>::zeros((2, 2 * d, ns)),
            n_samples: Array1::<u64>::zeros((2,)),

            d: d,
            ns: ns,
        }
    }
    /// Update the Ttest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    fn update(&mut self, py: Python, traces: PyReadonlyArray2<i16>, y: PyReadonlyArray1<u16>) {
        let traces = traces.as_array();
        let y = y.as_array();
        let cbs: Vec<(usize, Vec<(f64, usize)>)> = (2..((2 * self.d) + 1))
            .rev()
            .map(|j| {
                (
                    j,
                    (1..(j - 1)).map(|k| (binomial(j, k) as f64, k)).collect(),
                )
            })
            .collect();
        py.allow_threads(|| {
            traces
                .outer_iter()
                .zip(y.outer_iter())
                .for_each(|(traces, y)| {
                    let y = y.first().unwrap();

                    // update moments according to the value of y.
                    let mut n = self.n_samples.slice_mut(s![*y as usize]);
                    n += 1;
                    let n = *n.first().unwrap() as f64;
                    let mut cs = self.cs.slice_mut(s![*y as usize, .., ..]);
                    let mut m = self.m.slice_mut(s![*y as usize, ..]);

                    // compute delta
                    let delta = (traces.mapv(|t| t as f64) - &m) / (n as f64);
                    cbs.iter().for_each(|(j, vec)| {
                        if n > 1.0 {
                            let mult = (n - 1.0).powi(*j as i32)
                                * (1.0 - (-1.0 / (n - 1.0)).powi(*j as i32 - 1));
                            cs.slice_mut(s![*j - 1, ..])
                                .zip_mut_with(&delta, |r, delta| {
                                    *r += delta.powi(*j as i32) * mult
                                });
                        }
                        vec.iter().for_each(|(cb, k)| {
                            let i = (j - *k - 1)..(*j);
                            let tab = cs.slice_mut(s![i;*k, ..]);
                            let (a, mut b) = tab.split_at(Axis(0), 1);
                            Zip::from(&mut b.slice_mut(s![0, ..]))
                                .and(&a.slice(s![0, ..]))
                                .and(&delta)
                                .for_each(|dest, cs, delta| {
                                    *dest += cb * (-delta).powi(*k as i32) * cs
                                });
                        });
                    });

                    m += &delta;
                    cs.slice_mut(s![0, ..]).assign(&m);
                });
        });
    }
    fn get_ttest<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let mut ttest = Array2::<f64>::zeros((self.d, self.ns));
        let cs = &self.cs;
        let n_samples = &self.n_samples;

        let n0 = n_samples[[0]] as f64;
        let n1 = n_samples[[1]] as f64;

        let cm0 = &cs.slice(s![0,..,..])/(n_samples[[0]] as f64);
        let cm1 = &cs.slice(s![1,..,..])/(n_samples[[1]] as f64);
        
        let mut u0 = Array1::<f64>::zeros(self.ns);
        let mut u1 = Array1::<f64>::zeros(self.ns);

        let mut v0 = Array1::<f64>::zeros(self.ns);
        let mut v1 = Array1::<f64>::zeros(self.ns);
        py.allow_threads(|| {
            for d in 1..(self.d+1){
                if d == 1{
                    u0.assign(&self.m.slice(s![0,..]));
                    u1.assign(&self.m.slice(s![1,..]));
                    
                    v0.assign(&cm0.slice(s![1,..]));
                    v1.assign(&cm1.slice(s![1,..]));
                }else if d == 2{
                    u0.assign(&cm0.slice(s![1,..]));
                    u1.assign(&cm1.slice(s![1,..]));

                    Zip::from(&mut v0)
                        .and(&cm0.slice(s![3,..]))
                        .and(&cm0.slice(s![1,..]))
                        .for_each(|v,cmu,cml| *v = cmu - cml.powi(2));

                    Zip::from(&mut v1)
                        .and(&cm1.slice(s![3,..]))
                        .and(&cm1.slice(s![1,..]))
                        .for_each(|v,cmu,cml| *v = cmu - cml.powi(2));
                }else{
                    Zip::from(&mut u0)
                        .and(&cm0.slice(s![d-1,..]))
                        .and(&cm0.slice(s![1,..]))
                        .for_each(|v,cmu,cml| *v = cmu / cml.powf(d as f64/2.0));
                    Zip::from(&mut u1)
                        .and(&cm1.slice(s![d-1,..]))
                        .and(&cm1.slice(s![1,..]))
                        .for_each(|v,cmu,cml| *v = cmu / cml.powf(d as f64/2.0));

                    Zip::from(&mut v0)
                        .and(&cm0.slice(s![(d*2)-1,..]))
                        .and(&cm0.slice(s![d-1,..]))
                        .and(&cm0.slice(s![1,..]))
                        .for_each(|v,cmu,cml,cmd| *v = (cmu-cml.powi(2)) / cmd.powi(d as i32));
                    Zip::from(&mut v1)
                        .and(&cm1.slice(s![(d*2)-1,..]))
                        .and(&cm1.slice(s![d-1,..]))
                        .and(&cm1.slice(s![1,..]))
                        .for_each(|v,cmu,cml,cmd| *v = (cmu-cml.powi(2)) / cmd.powi(d as i32));
                }

                Zip::from(&mut ttest.slice_mut(s![d-1,..])).
                    and(&u0).and(&u1).and(&v0).and(&v1).for_each(|t,u0,u1,v0,v1|{
                        *t = (u0-u1)/f64::sqrt((v0/n0) + (v1/n1));
                    }
                );
            }

        });
        Ok(&(ttest.to_pyarray(py)))
    }

}
