//! Estimation for higher-order Univariate T-test.
//!
//! An estimation of Ttest is represented with a Ttest struct. Calling update allows
//! to update the Ttest state with fresh measurements. get_ttest returns the current value
//! of the estimate.
//! The measurements are expected to be of length ns.
//!
//! This is based on the one-pass algorithm proposed in
//! <https://eprint.iacr.org/2015/207>.
use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, Axis};
use num_integer::binomial;
use rayon::prelude::*;
use std::cmp;
const NS_BATCH: usize = 1 << 12;
const Y_BATCH: usize = 1 << 9;

pub struct UniCSAcc {
    /// Number of samples in trace
    pub ns: usize,
    /// pub Number of classes
    pub nc: usize,
    /// Highest moment to estimate
    pub d: usize,
    /// Number of samples in sets
    pub n_traces: Array1<u64>,
    /// Array containing the current estimation
    pub moments: Array3<f64>,
}

impl UniCSAcc {
    /// Creates an UniCSAcc
    /// ns : number of point in traces
    /// d : higher power to estimate
    /// nc: number of classes to estimate the CS for.
    pub fn new(ns: usize, d: usize, nc: usize) -> Self {
        UniCSAcc {
            ns: ns,
            d: d,
            nc: nc,
            n_traces: Array1::<u64>::zeros(nc),
            moments: Array3::<f64>::zeros((nc, d, ns)),
        }
    }
    /// Merges the current accumulator with anoter sum of
    /// centered products
    /// moments_other : (nc,d,ns) matrix
    /// n_traces : (nc,) matrix
    // See
    // [1]: https://www.osti.gov/biblio/1028931-formulas-robust-one-pass-parallel-computation-covariances-arbitrary-order-statistical-moments
    // [2]: https://eprint.iacr.org/2015/207.pdf
    // for additional details about the merging algorithm.
    //
    // Q0 set of all the traces in current estimation
    // Q1 set of all the traces in the estiamation to merge
    // Q = Q0 U Q1 is the merge of all traces
    //
    // Definitions:
    //      M_d,Qi = sum_{Qi} (l - u_i) ^ d
    //      u_Qi = 1/(|Qi|) * sum_{Qi} l
    //      delta_1,0 = u_0 - u_1
    //      ni = |Qi|
    //      n = n0 + n1
    //
    // Update equation (Eq. 2.1 in [1]):
    //
    // M_Q_d = M_Q0_d + M_Q1_d +
    //      sum_{k=1}_{p-2}(
    //           binomial(k,p) *
    //
    //              delta_1,0 ^ d *
    //              (
    //                  ((-n1/n) ^ k * M_p-k,0) +
    //                   (n0/n) ^ k * M_p-k,1))
    //              )
    //      )
    //      + ((n0 * n1) / n ) *( delta_1,0 ^ p)
    //              * ( 1 / (n1 ^ (p-1)) - (-1 / n0)^(p-1))
    pub fn merge_from_state(&mut self, moments_other: ArrayView3<f64>, n_traces: ArrayView1<u64>) {
        // compute the deltas u1 - u0 for all the classes
        let delta = &moments_other.slice(s![.., 0, ..]) - &self.moments.slice(s![.., 0, ..]);
        let d = self.d;
        let ns = self.ns;

        // each row will contain one power of deltas
        let mut delta_pows = Array2::<f64>::zeros((d + 1, ns));

        // Update all the classes estimated independently
        izip!(
            &mut self.moments.outer_iter_mut(),
            self.n_traces.iter(),
            moments_other.outer_iter(),
            n_traces.iter(),
            delta.outer_iter()
        )
        .for_each(|(mut cs0, n0, cs1, n1, delta)| {
            let n = (*n0 + *n1) as f64;
            let n0 = *n0 as f64;
            let n1 = *n1 as f64;

            if n0 == 0.0 {
                // if current estimate has no sample, simply assign from the other estimate.
                cs0.assign(&cs1);
            } else if n1 != 0.0 {
                // compute powers of deltas
                delta_pows
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(i, mut x)| x.assign(&delta.mapv(|t| t.powi(i as i32))));

                // first update the higher powers since their update rules
                // is based on the one of smaller powers.
                for p in (2..d + 1).rev() {
                    let p = p as i32;

                    let (as_input0, mut to_update0) =
                        cs0.view_mut().split_at(Axis(0), (p - 1) as usize);
                    let (as_input1, to_update1) = cs1.view().split_at(Axis(0), (p - 1) as usize);

                    let mut to_update0 = to_update0.slice_mut(s![0, ..]);
                    to_update0 += &to_update1.slice(s![0, ..]);

                    for k in 1..(p - 1) {
                        let cst = binomial(p, k) as f64;
                        let mut tmp = &delta_pows.slice(s![k, ..]) * cst;
                        let tmp2 = &as_input0.slice(s![p - k - 1, ..]) * ((-n1 / n).powi(k));
                        let tmp3 = &as_input1.slice(s![p - k - 1, ..]) * ((n0 / n).powi(k));
                        let x = tmp2 + tmp3;
                        tmp = &tmp * x;
                        to_update0 += &tmp;
                    }

                    let mut cst = (1.0 / n1).powi(p - 1) - (-1.0 / n0).powi(p - 1);
                    cst *= (n1 * n0 / n).powi(p);
                    to_update0 += &(&delta_pows.slice(s![p, ..]) * cst);
                }

                // update the mean
                let mut u0 = cs0.slice_mut(s![0, ..]);
                u0 += &(&delta * (n1 / n));
            }
        });
        self.n_traces += &n_traces;
    }

    /// Merges to different CS estimations.
    pub fn merge(&mut self, other: &Self) {
        self.merge_from_state(other.moments.view(), other.n_traces.view());
    }

    /// Updates the current estimation with fresh traces.
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let mut sum = Array2::<i64>::zeros((self.nc, self.ns));
        let mut moments_other = Array3::<f64>::zeros((self.nc, self.d, self.ns));
        let mut n_traces = Array1::<u64>::zeros(self.nc);

        // STEP 1: 2-passes algorithm to compute center sum of powers
        // sum per class,
        for (trace, class) in traces.outer_iter().zip(y.iter()) {
            n_traces[*class as usize] += 1;
            let mut s = sum.slice_mut(s![*class as usize, ..]);
            acc_sum(&mut s.view_mut(), &trace);
        }

        // assign mean
        let n = n_traces.mapv(|x| x as f64);
        let mut mean = sum.mapv(|x| x as f64);
        mean.axis_iter_mut(Axis(1)).for_each(|mut m| m /= &n);
        moments_other.slice_mut(s![.., 0, ..]).assign(&mean);

        // compute centered sums
        let mut pow = Array1::<f64>::zeros((self.ns,));
        let mut t = Array1::<f64>::zeros((self.ns,));
        for (trace, class) in traces.outer_iter().zip(y.iter()) {
            let mut m_full = moments_other.slice_mut(s![*class as usize, .., ..]);
            izip!(
                t.view_mut().into_slice().unwrap().iter_mut(),
                pow.view_mut().into_slice().unwrap().iter_mut(),
                m_full.slice(s![0, ..]).view().to_slice().unwrap().iter(),
                trace.view().to_slice().unwrap().iter()
            )
            .for_each(|(t, pow, m_full, trace)| {
                *t = *trace as f64 - m_full;
                *pow = *t;
            });
            for d in 2..(self.d + 1) {
                let dest = m_full.slice_mut(s![d - 1, ..]);
                pow_and_sum(dest, pow.view_mut(), t.view());
            }
        }

        // STEP 2 merge the two pass estimation with the current state of the accumulator
        self.merge_from_state(moments_other.view(), n_traces.view());
    }

    /// Reset the current accumulator
    pub fn reset(&mut self) {
        self.n_traces.fill(0);
        self.moments.fill(0.0);
    }
}
pub struct Ttest {
    /// order of the test
    d: usize,
    /// Number of samples per trace
    ns: usize,
    /// Vector of Moment accumulators
    accumulators: Vec<UniCSAcc>,
}

pub fn build_accumulator(ns: usize, d: usize) -> Vec<UniCSAcc> {
    let n_batches = ((ns as f64) / (NS_BATCH as f64)).ceil() as usize;
    let accumulators: Vec<UniCSAcc> = (0..n_batches)
        .map(|x| {
            let l = std::cmp::min(ns - (x * NS_BATCH), NS_BATCH);
            UniCSAcc::new(l, 2 * d, 2)
        })
        .collect();
    accumulators
}
impl Ttest {
    /// Create a new Ttest state.
    /// ns: traces length
    /// d: order of the Ttest
    pub fn new(ns: usize, d: usize) -> Self {
        // number of required accumulators
        let accumulators = build_accumulator(ns, d);
        Ttest {
            d: d,
            ns: ns,
            accumulators: accumulators,
        }
    }
    /// Update the Ttest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let d = self.d;
        let ns = self.ns;
        let n_traces = traces.shape()[0];
        let ns_chuncks = cmp::max(1, ns / NS_BATCH);
        let min_desired_chuncks = 4 * rayon::current_num_threads();

        let y_chunck_size = if min_desired_chuncks < ns_chuncks {
            n_traces
        } else {
            let tmp = cmp::min(
                rayon::current_num_threads(),
                min_desired_chuncks / ns_chuncks,
            ); // ensure that we do not split in more than available threads.
            cmp::max(256, n_traces / tmp)
        };

        let res = (
            traces.axis_chunks_iter(Axis(0), y_chunck_size),
            y.axis_chunks_iter(Axis(0), y_chunck_size),
        )
            .into_par_iter()
            .map(|(traces, y)| {
                // chunck different traces for more threads
                let mut accumulators = build_accumulator(ns, d);
                (
                    traces.axis_chunks_iter(Axis(1), NS_BATCH),
                    &mut accumulators,
                )
                    .into_par_iter()
                    .for_each(|(traces, acc)| {
                        // chunck the traces with their lenght
                        izip!(
                            traces.axis_chunks_iter(Axis(0), Y_BATCH),
                            y.axis_chunks_iter(Axis(0), Y_BATCH)
                        )
                        .for_each(|(traces, y)| acc.update(traces, y));
                    });
                accumulators
            })
            .reduce(
                || build_accumulator(ns, d),
                |mut x, y| {
                    // accumulate all to the self accumulator
                    x.iter_mut().zip(y.iter()).for_each(|(x, y)| x.merge(y));
                    x
                },
            );
        izip!(self.accumulators.iter_mut(), res.iter()).for_each(|(x, y)| x.merge(y));
    }

    /// Generate the actual Ttest metric based on the current state.
    /// return array axes (d,ns)
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
    pub fn get_ttest(&self) -> Array2<f64> {
        let mut ttest = Array2::<f64>::zeros((self.d, self.ns));
        let n_samples = &self.accumulators[0].n_traces;

        let n0 = n_samples[[0]] as f64;
        let n1 = n_samples[[1]] as f64;

        izip!(
            ttest.axis_chunks_iter_mut(Axis(1), NS_BATCH),
            self.accumulators.iter()
        )
        .for_each(|(mut ttest, acc)| {
            ttest
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(d, mut ttest)| {
                    let d = d + 1;

                    if d == 1 {
                        let u0 = acc.moments.slice(s![0, 0, ..]);
                        let u1 = acc.moments.slice(s![1, 0, ..]);
                        let v0 = acc.moments.slice(s![0, 1, ..]);
                        let v0 = &v0 / n0;
                        let v1 = acc.moments.slice(s![1, 1, ..]);
                        let v1 = &v1 / n1;
                        let t = (&u0 - &u1)
                            / (&v0.mapv(|x| x / n0) + &v1.mapv(|x| x / n1)).mapv(|x| f64::sqrt(x));
                        ttest.assign(&t);
                    } else if d == 2 {
                        let u0 = acc.moments.slice(s![0, 1, ..]).mapv(|x| x / n0);
                        let u1 = acc.moments.slice(s![1, 1, ..]).mapv(|x| x / n1);

                        let v0 = acc.moments.slice(s![0, 3, ..]).mapv(|x| x / n0);
                        let v1 = acc.moments.slice(s![1, 3, ..]).mapv(|x| x / n1);
                        let v0 = &v0 - &u0.mapv(|x| x.powi(2));
                        let v1 = &v1 - &u1.mapv(|x| x.powi(2));
                        let t = (&u0 - &u1)
                            / (&v0.mapv(|x| x / n0) + &v1.mapv(|x| x / n1)).mapv(|x| f64::sqrt(x));
                        ttest.assign(&t);
                    } else {
                        let u0 = &(acc.moments.slice(s![0, d - 1, ..])).mapv(|x| x / n0)
                            / &(acc
                                .moments
                                .slice(s![0, 1, ..])
                                .mapv(|x| (x / n0).powf(d as f64 / 2.0)));

                        let u1 = &(acc.moments.slice(s![1, d - 1, ..])).mapv(|x| x / n1)
                            / &(acc
                                .moments
                                .slice(s![1, 1, ..])
                                .mapv(|x| (x / n1).powf(d as f64 / 2.0)));

                        let v0 = (&acc.moments.slice(s![0, (2 * d) - 1, ..]).mapv(|x| x / n0)
                            - &acc
                                .moments
                                .slice(s![0, d - 1, ..])
                                .mapv(|x| (x / n0).powi(2)))
                            / &(acc
                                .moments
                                .slice(s![0, 1, ..])
                                .mapv(|x| (x / n0).powi(d as i32)));

                        let v1 = (&acc.moments.slice(s![1, (2 * d) - 1, ..]).mapv(|x| x / n1)
                            - &acc
                                .moments
                                .slice(s![1, d - 1, ..])
                                .mapv(|x| (x / n1).powi(2)))
                            / &(acc
                                .moments
                                .slice(s![1, 1, ..])
                                .mapv(|x| (x / n1).powi(d as i32)));
                        let t = (&u0 - &u1)
                            / (&v0.mapv(|x| x / n0) + &v1.mapv(|x| x / n1)).mapv(|x| f64::sqrt(x));
                        ttest.assign(&t);
                    }
                });
        });
        return ttest;
    }
}

#[inline(never)]
pub fn prod_update(vec: &mut [f64], pv: &[f64], d: &[f64], cst: f64) {
    izip!(vec.iter_mut(), pv.into_iter(), d.into_iter()).for_each(|(v, p, d)| *v += cst * *p * *d);
}
#[inline(never)]
pub fn prod_update_add(vec: &mut [f64], pv: &[f64], cst: f64) {
    izip!(vec.iter_mut(), pv.into_iter()).for_each(|(v, p)| *v += cst * *p);
}
#[inline(never)]
pub fn prod_update_single(vec: &mut [f64], pv: &[f64], d: &[f64]) {
    izip!(vec.iter_mut(), pv.into_iter(), d.into_iter()).for_each(|(v, p, d)| *v = *p * *d);
}

#[inline(never)]
pub fn gen_delta(m: &mut [f64], d: &mut [f64], t: &[i16], poi: &[u64], n: f64) {
    izip!(m.iter_mut(), d.iter_mut())
        .enumerate()
        .for_each(|(i, (m, d))| {
            *d = t[poi[i as usize] as usize] as f64 - *m;
            *m += (*d) / (n);
        });
}

#[inline(never)]
pub fn acc_sum(m: &mut ArrayViewMut1<i64>, t: &ArrayView1<i16>) {
    m.zip_mut_with(t, |m, t| *m += *t as i64);
}

#[inline(always)]
pub fn pow_and_sum(m: ArrayViewMut1<f64>, pow: ArrayViewMut1<f64>, t: ArrayView1<f64>) {
    let m = m.into_slice().unwrap();
    let pow = pow.into_slice().unwrap();
    let t = t.to_slice().unwrap();
    izip!(m.iter_mut(), pow.iter_mut(), t.iter()).for_each(|(m, pow, t)| {
        *pow *= *t;
        *m += *pow;
    });
}
