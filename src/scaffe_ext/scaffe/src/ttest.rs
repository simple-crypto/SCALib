//! Estimation for higher-order T-test.
//!
//! An estimation of Ttest is represented with a Ttest struct. Calling update allows
//! to update the Ttest state with fresh measurements. get_ttest returns the current value
//! of the estimate.
//! The measurements are expected to be of length ns.
//!
//! This is based on the one-pass algorithm proposed in
//! <https://eprint.iacr.org/2015/207>.

use ndarray::{s, Array1, Array2, Array3, Axis};
use num_integer::binomial;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct Ttest {
    /// Central sums of order 1 up to order d*2 with shape (ns,2,2*d),
    /// where central sums is sum((x-u_x)**i).
    /// Axes are (class, trace sample, order).
    /// cs[..,..,0] contains the current estimation of means instead of
    /// the central sum (which would be zero).
    /// number of samples in a trace
    cs: Array3<f64>,
    /// number of samples per class (2,)
    n_samples: Array1<u64>,
    /// order of the test
    d: usize,
    /// Number of samples per trace
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
            cs: Array3::<f64>::zeros((ns, 2, 2 * d)),
            n_samples: Array1::<u64>::zeros((2,)),
            d: d,
            ns: ns,
        }
    }
    /// Update the Ttest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    // Q set of all previous traces
    //
    // Initial values, |Q| = n-1
    // mu = (1/(n-1)) * sum(Q)
    // if d>=2:
    //      CS_{d,Q} = 1/(n-1) sum((Q-mu)**d)
    // else:
    //      CS_{1,Q'} = mu'
    //
    // Updated with a single measurement t, Q' = Q U t, |Q'| = n
    // mu' = (1/(n-1)) * sum(Q')
    //
    // if d >=2:
    //      CS_{d,Q'} = (1/n) sum((Q-mu')**d)
    // else:
    //      CS_{1,Q'} = mu'
    //
    // Update rules is given by:
    //
    // delta = (t - mu)/n
    //
    // CS_{d,Q'} = CS_{d,Q}
    //      + sum_{k=1,d-2}(
    //         binomial(k,d)
    //         * CS_{d-k,Q}
    //         * (-delta)**d
    //         )
    //      + (
    //          (delta * (n-1))**d
    //          *(1 - (-1/(n-1)))**(d-1)
    //        )
    //
    // mu' = mu+delta
    fn update(&mut self, py: Python, traces: PyReadonlyArray2<i16>, y: PyReadonlyArray1<u16>) {
        let traces = traces.as_array();
        let y = y.as_array();
        py.allow_threads(|| {
            let d = self.d;

            // pre computes the combinatorial factors
            let cbs: Vec<(usize, Vec<(f64, usize)>)> = (2..((2 * self.d) + 1))
                .rev()
                .map(|j| {
                    (
                        j,
                        (1..(j - 1)).map(|k| (binomial(j, k) as f64, k)).collect(),
                    )
                })
                .collect();

            // contains the data that are the same for all the points in a single traces
            // Contains tupes (n, y, mults): (f64, usize, Vec<f64>)
            // n : number of previously processed traces for the class y
            // y : set to update
            // mults: (n-1)**(j) * (1.0 - (-1.0/(n-1.0))**(j-1) for j in [2..(2*d-1)].rev()
            let shared_data: Vec<(f64, usize, Vec<f64>)> = y
                .iter()
                .map(|y| {
                    let y = *y as usize;
                    assert!(y <= 1);

                    // update the number of observations
                    let n = &mut self.n_samples[y];
                    *n += 1;
                    let n = *n as f64;

                    (
                        // number of sample on that class
                        n,
                        // y value
                        y,
                        // compute the multiplicative factor similar for all trace samples
                        cbs.iter()
                            .map(|(j, _)| {
                                (n - 1.0).powi(*j as i32)
                                    * (1.0 - (-1.0 / (n - 1.0)).powi(*j as i32 - 1))
                            })
                            .collect(),
                    )
                })
                .collect();

            (traces.axis_iter(Axis(1)), self.cs.axis_iter_mut(Axis(0)))
                .into_par_iter()
                .for_each_init(
                    || Array1::<f64>::zeros(2 * d),
                    |ref mut delta_pows, (traces, mut cs)| {
                        traces
                            .iter()
                            .zip(shared_data.iter())
                            .for_each(|(trace, (n, y, mults))| {
                                let mut cs_s = cs.slice_mut(s![*y, ..]);
                                let cs = cs_s.as_slice_mut().unwrap();

                                // compute the delta
                                let delta = ((*trace as f64) - cs[0]) / (*n as f64);

                                // delta_pows[i] = delta ** (i+1)
                                // We will need all of them next
                                delta_pows.iter_mut().fold(delta, |acc, x| {
                                    *x = acc;
                                    acc * delta
                                });

                                // apply the one-pass update rule
                                cbs.iter().zip(mults.iter()).for_each(|((j, vec), mult)| {
                                    if *n > 1.0 {
                                        cs[*j - 1] += delta_pows[*j - 1] * mult;
                                    }
                                    vec.iter().for_each(|(cb, k)| {
                                        let a = cs[*j - *k - 1];
                                        if (k & 0x1) == 1 {
                                            // k is not pair
                                            cs[*j - 1] -= cb * delta_pows[*k - 1] * a;
                                        } else {
                                            // k is pair
                                            cs[*j - 1] += cb * delta_pows[*k - 1] * a;
                                        }
                                    });
                                });
                                cs[0] += delta;
                            });
                    },
                );
        });
    }

    /// Generate the actual Ttest metric based on the current state.
    /// return array axes (d,ns)
    //
    // with central moment defined as:
    //   CM_{i,Q} = CS_{i,Q}/n
    //
    // the statistic is given by:
    // t = (u0 - u1) / sqrt( s0/n0 + s1/n1)
    //
    // d = 1:
    //      ui = sum(t)/ni
    //      vi = CM_{2,Q}
    // d = 2:
    //      ui = CM_{2,Q}
    //      vi = CM_{4,Q} - CM_{2,Q}**2
    // d > 2:
    //      ui = CM_{d,Q} / CM_{2,Q}**(d/2)
    //      vi = (CM_{2*d,Q} - CM_{d,Q}**2) / CM{2,Q}**d

    fn get_ttest<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let mut ttest = Array2::<f64>::zeros((self.d, self.ns));
        let cs = &self.cs;
        let n_samples = &self.n_samples;

        let n0 = n_samples[[0]] as f64;
        let n1 = n_samples[[1]] as f64;

        py.allow_threads(|| {
            (
                ttest.axis_chunks_iter_mut(Axis(1), 20),
                cs.axis_chunks_iter(Axis(0), 20),
            )
                .into_par_iter()
                .for_each(|(mut ttest, cs)| {
                    ttest
                        .axis_iter_mut(Axis(1))
                        .zip(cs.axis_iter(Axis(0)))
                        .for_each(|(mut ttest, cs)| {
                            let mut u0;
                            let mut u1;
                            let mut v0;
                            let mut v1;
                            for d in 1..(self.d + 1) {
                                if d == 1 {
                                    u0 = cs[[0, 0]];
                                    u1 = cs[[1, 0]];

                                    v0 = cs[[0, 1]] / n0;
                                    v1 = cs[[1, 1]] / n1;
                                } else if d == 2 {
                                    u0 = cs[[0, 1]] / n0;
                                    u1 = cs[[1, 1]] / n1;

                                    v0 = cs[[0, 3]] / n0 - ((cs[[0, 1]] / n0).powi(2));
                                    v1 = cs[[1, 3]] / n1 - ((cs[[1, 1]] / n1).powi(2));
                                } else {
                                    u0 = (cs[[0, d - 1]] / n0)
                                        / ((cs[[0, 1]] / n0).powf(d as f64 / 2.0));
                                    u1 = (cs[[1, d - 1]] / n1)
                                        / ((cs[[1, 1]] / n1).powf(d as f64 / 2.0));

                                    v0 =
                                        cs[[0, (2 * d) - 1]] / n0 - ((cs[[0, d - 1]] / n0).powi(2));
                                    v0 /= (cs[[0, 1]] / n0).powi(d as i32);

                                    v1 =
                                        cs[[1, (2 * d) - 1]] / n1 - ((cs[[1, d - 1]] / n1).powi(2));
                                    v1 /= (cs[[1, 1]] / n1).powi(d as i32);
                                }

                                ttest[d - 1] = (u0 - u1) / f64::sqrt((v0 / n0) + (v1 / n1));
                            }
                        });
                });
        });
        Ok(&(ttest.to_pyarray(py)))
    }
}
