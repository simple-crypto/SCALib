//! Estimation for higher-order T-test.
//!
//! An estimation of Ttest is represented with a Ttest struct. Calling update allows
//! to update the Ttest state with fresh measurements. get_ttest returns the current value
//! of the estimate.
//! The measurements are expected to be of length ns.
//!
//! This is based on the one-pass algorithm proposed in
//! <https://eprint.iacr.org/2015/207>.

use itertools::{izip, Itertools};
use ndarray::{
    s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2,
    ArrayViewMut3, Axis,
};
use num_integer::binomial;
use rayon::prelude::*;

pub struct UnivarMomentAcc {
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

impl UnivarMomentAcc {
    pub fn new(ns: usize, d: usize, nc: usize) -> Self {
        UnivarMomentAcc {
            ns: ns,
            d: d,
            nc: nc,
            n_traces: Array1::<u64>::zeros(nc),
            moments: Array3::<f64>::zeros((nc, d, ns)),
        }
    }
    pub fn merge_from_state(&mut self, moments_other: ArrayView3<f64>, n_traces: ArrayView1<u64>) {
        let delta = &moments_other.slice(s![.., 0, ..]) - &self.moments.slice(s![.., 0, ..]);
        let d = self.d;
        let ns = self.ns;
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
                cs0.assign(&cs1);
            } else if n1 != 0.0 {
                let mut delta_pows = Array2::<f64>::zeros((d + 1, ns));
                delta_pows
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(i, mut x)| x.assign(&delta.mapv(|t| t.powi(i as i32))));
                for p in (2..d + 1).rev() {
                    let p = p as i32;
                    let (as_input0, mut to_update0) =
                        cs0.view_mut().split_at(Axis(0), (p - 1) as usize);
                    let (as_input1, to_update1) = cs1.view().split_at(Axis(0), (p - 1) as usize);
                    let mut to_update0 = to_update0.slice_mut(s![0, ..]);
                    let to_update1 = to_update1.slice(s![0, ..]);
                    to_update0 += &to_update1;

                    let mut cst = (1.0 / n1).powi(p - 1) - (-1.0 / n0).powi(p - 1);
                    cst *= (n1 * n0 / n).powi(p);
                    to_update0 += &(&delta_pows.slice(s![p, ..]) * cst);
                    for k in 1..(p - 1) {
                        let cst = binomial(p, k) as f64;
                        let mut tmp = &delta_pows.slice(s![k, ..]) * cst;
                        let tmp2 = &as_input0.slice(s![p - k - 1, ..]) * ((-n1 / n).powi(k));
                        let tmp3 = &as_input1.slice(s![p - k - 1, ..]) * ((n0 / n).powi(k));
                        let x = tmp2 + tmp3;
                        tmp = &tmp * x;
                        to_update0 += &tmp;
                    }
                }
                let mut u0 = cs0.slice_mut(s![0, ..]);
                u0 += &(&delta * (n1 / n));
            }
        });
        self.n_traces += &n_traces;
    }
    pub fn merge(&mut self, other: &Self) {
        self.merge_from_state(other.moments.view(), other.n_traces.view());
    }
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let mut sum = Array2::<u64>::zeros((self.nc, self.ns));
        let mut moments_other = Array3::<f64>::zeros((self.nc, self.d, self.ns));
        let mut n_traces = Array1::<u64>::zeros(self.nc);

        // STEP 1: 2-passes algorithm to compute higher-order moments
        // sum and sum of square per class
        for (trace, class) in traces.outer_iter().zip(y.iter()) {
            n_traces[*class as usize] += 1;
            let mut s = sum.slice_mut(s![*class as usize, ..]);
            s.zip_mut_with(&trace, |s, t| {
                *s += *t as u64;
            });
        }

        // assign mean
        let n = n_traces.mapv(|x| x as f64);
        let mut mean = sum.mapv(|x| x as f64);
        mean.axis_iter_mut(Axis(1)).for_each(|mut m| m /= &n);
        moments_other.slice_mut(s![.., 0, ..]).assign(&mean);

        // compute centered sums
        for (trace, class) in traces.outer_iter().zip(y.iter()) {
            let mut m_full = moments_other.slice_mut(s![*class as usize, .., ..]);
            for d in 2..(self.d + 1) {
                // centering
                let mut m = &trace.mapv(|x| x as f64) - &m_full.slice(s![0, ..]);
                m.mapv_inplace(|x| x.powi(d as i32));
                let mut dest = m_full.slice_mut(s![d - 1, ..]);
                dest += &m;
            }
        }

        // STEP 2 merge in current state
        self.merge_from_state(moments_other.view(), n_traces.view());
    }

    pub fn reset(&mut self) {
        self.n_traces.fill(0);
        self.moments.fill(0.0);
    }
}
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

impl Ttest {
    /// Create a new Ttest state.
    /// ns: traces length
    /// d: order of the Ttest
    pub fn new(ns: usize, d: usize) -> Self {
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
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
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

        let mut delta_pows = Array1::<f64>::zeros(2 * d);
        izip!(traces.axis_iter(Axis(1)), self.cs.axis_iter_mut(Axis(0))).for_each(
            |(traces, mut cs)| {
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

    pub fn get_ttest(&self) -> Array2<f64> {
        let mut ttest = Array2::<f64>::zeros((self.d, self.ns));
        let cs = &self.cs;
        let n_samples = &self.n_samples;

        let n0 = n_samples[[0]] as f64;
        let n1 = n_samples[[1]] as f64;

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

                                v0 = cs[[0, (2 * d) - 1]] / n0 - ((cs[[0, d - 1]] / n0).powi(2));
                                v0 /= (cs[[0, 1]] / n0).powi(d as i32);

                                v1 = cs[[1, (2 * d) - 1]] / n1 - ((cs[[1, d - 1]] / n1).powi(2));
                                v1 /= (cs[[1, 1]] / n1).powi(d as i32);
                            }

                            ttest[d - 1] = (u0 - u1) / f64::sqrt((v0 / n0) + (v1 / n1));
                        }
                    });
            });
        return ttest;
    }
}

pub struct MTtest {
    /// Central sums of order 1 up to order d*2 with shape (ns,2,2*d),
    /// where central sums is sum((x-u_x)**i).
    /// Axes are (class, trace sample, order).
    /// cs[..,..,0] contains the current estimation of means instead of
    /// the central sum (which would be zero).

    /// number of samples per class (2,)
    n_samples: Array1<u64>,
    /// Current state of all combinations of POIs
    states: Vec<Vec<(Vec<usize>, usize)>>,
    states_plain: Array3<f64>,
    /// Vector containg the delta's for all combinations of POIS
    delta_prods: Vec<Vec<(Vec<usize>, usize, usize, usize)>>,
    delta_prods_plain: Array2<f64>,

    /// POIS
    pois: Array2<u64>,
    /// Central first of moments
    m: Array3<f64>,
    /// Number of shares
    d: usize,
    /// Number of samples in traces
    ns: usize,
    /// Represents all the equations that must be computed to update a state
    equations: Vec<Vec<(usize, usize, Vec<(usize, usize, usize, usize, f64)>)>>,
}

impl MTtest {
    /// Create a new Ttest state.
    /// ns: traces length
    /// d: orde of the Ttest
    pub fn new(d: usize, pois: ArrayView2<u64>) -> Self {
        let ns = pois.shape()[1];
        assert!(d == pois.shape()[0]);

        // the set that must be computed
        let sets = Array::range(0.0, 2.0 * d as f64, 1.0).map(|x| (*x as usize) % d);

        // for all size of combinations, generate the all unique combinations. For each of them,
        // initialize and array that will maintain the current estimate.
        let mut states_cnt = 0;
        let states: Vec<Vec<(Vec<usize>, usize)>> = (2..(2 * d + 1))
            .map(|l| {
                let mut tmp: Vec<Vec<usize>> = sets.clone().into_iter().combinations(l).collect();

                tmp.iter_mut().for_each(|x| x.sort());
                let combi_single: Vec<Vec<usize>> = tmp.clone().into_iter().unique().collect();
                combi_single
                    .into_iter()
                    .map(|x| {
                        states_cnt = states_cnt + 1;
                        (x, states_cnt - 1)
                    })
                    .collect()
            })
            .collect();

        // for all size of combinations, generate the all unique combinations. For each of them,
        // initialize and array that will maintain the current estimate.
        let mut delta_cnt = 0;
        let mut delta_prods: Vec<Vec<(Vec<usize>, usize, usize, usize)>> = (1..(2 * d + 1))
            .map(|l| {
                let mut tmp: Vec<Vec<usize>> = sets.clone().into_iter().combinations(l).collect();
                tmp.iter_mut().for_each(|x| x.sort());
                let combi_single: Vec<Vec<usize>> = tmp.clone().into_iter().unique().collect();
                combi_single
                    .into_iter()
                    .map(|x| {
                        delta_cnt += 1;
                        (x, delta_cnt - 1, 0, 0)
                    })
                    .collect()
            })
            .collect();

        // derive the index that are use to update deltas
        for size in 2..(2 * d + 1) {
            // size of the delta to update
            let (as_input, to_updates) = delta_prods.split_at_mut(size - 1);

            to_updates[0].iter_mut().for_each(|to_update| {
                let combi = &to_update.0;
                let (low, up) = combi.split_at(size / 2);
                let low_data: usize = as_input[low.len() - 1]
                    .iter()
                    .filter(|x| x.0 == low)
                    .map(|x| x.1)
                    .collect::<Vec<usize>>()[0];
                let up_data: usize = as_input[up.len() - 1]
                    .iter()
                    .filter(|x| x.0 == up)
                    .map(|x| x.1)
                    .collect::<Vec<usize>>()[0];
                to_update.2 = low_data;
                to_update.3 = up_data;
            });
        }

        let states_plain = Array3::<f64>::zeros((2, states_cnt, ns));
        let delta_prods_plain = Array2::<f64>::zeros((delta_cnt, ns));

        // for all the states to update, generate all the equations.
        let equations: Vec<Vec<(usize, usize, Vec<(usize, usize, usize, usize, f64)>)>> = states
            .iter()
            .map(|state| {
                state
                    .iter()
                    .map(|(combi, _)| {
                        let mixed_eq: Vec<(usize, usize, usize, usize, f64)> = (2..combi.len())
                            .map(|size| {
                                let mut sub_combi: Vec<Vec<usize>> = combi
                                    .clone()
                                    .into_iter()
                                    .combinations(size)
                                    .collect::<Vec<Vec<usize>>>();
                                sub_combi.iter_mut().for_each(|x| x.sort());
                                let sub_combi_unique: Vec<Vec<usize>> =
                                    sub_combi.clone().into_iter().unique().collect();
                                let counts: Vec<usize> = sub_combi_unique
                                    .clone()
                                    .iter()
                                    .map(|x| {
                                        sub_combi.clone().into_iter().filter(|y| *x == *y).count()
                                    })
                                    .collect();
                                let missing: Vec<Vec<usize>> = sub_combi_unique
                                    .clone()
                                    .into_iter()
                                    .map(|sub_combi| {
                                        let mut x = combi.clone();
                                        for i in sub_combi.iter() {
                                            let posi = x.iter().position(|x| *x == *i).unwrap();
                                            x.remove(posi);
                                        }
                                        x
                                    })
                                    .collect();
                                izip!(
                                    sub_combi_unique.into_iter(),
                                    missing.into_iter(),
                                    counts.into_iter()
                                )
                                .map(|(sc, c, count)| {
                                    // position of sc in states
                                    let states_posi_0 = sc.len() - 2;
                                    let states_posi_1 = states[states_posi_0]
                                        .iter()
                                        .position(|x| x.0 == sc)
                                        .unwrap();
                                    // position in delta_prods
                                    let delta_posi_0 = c.len() - 1;
                                    let delta_posi_1 = delta_prods[delta_posi_0]
                                        .iter()
                                        .position(|x| x.0 == c)
                                        .unwrap();
                                    // occurence of sub_combi in combi
                                    (
                                        states_posi_0,
                                        states_posi_1,
                                        delta_posi_0,
                                        delta_posi_1,
                                        (-1.0_f64).powi(c.len() as i32) * count as f64,
                                    )
                                })
                                .collect::<Vec<(
                                    usize,
                                    usize, // position in state
                                    usize, // combination in delta_prods
                                    usize,
                                    f64,
                                )>>(
                                )
                            })
                            .flatten()
                            .collect();
                        let posi_c = delta_prods[combi.len() - 1]
                            .iter()
                            .position(|x| x.0 == *combi)
                            .unwrap();
                        (combi.len() - 1, posi_c, mixed_eq)
                    })
                    .collect()
            })
            .collect();

        MTtest {
            n_samples: Array1::<u64>::zeros((2,)),
            pois: pois.to_owned(),
            states: states,
            states_plain: states_plain,
            delta_prods: delta_prods,
            delta_prods_plain: delta_prods_plain,
            equations: equations,
            m: Array3::<f64>::zeros((d, 2, ns)),
            d: d,
            ns: ns,
        }
    }

    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let dims = traces.shape();

        // compute the updates n_samples
        let mut n_evol = Array1::<f64>::zeros((dims[0],));
        n_evol.iter_mut().zip(y.iter()).for_each(|(evol, y)| {
            let n = &mut self.n_samples[*y as usize];
            *n += 1;
            *evol = *n as f64;
        });

        let csize = 1 << 10;
        let delta_prods = &self.delta_prods;
        let states = &self.states;
        let equations = &self.equations;
        let d = self.d;
        (
            self.pois.axis_chunks_iter(Axis(1), csize),
            self.delta_prods_plain.axis_chunks_iter_mut(Axis(1), csize),
            self.states_plain.axis_chunks_iter_mut(Axis(2), csize),
            self.m.axis_chunks_iter_mut(Axis(2), csize),
        )
            .into_par_iter()
            .for_each(|(pois, dpp, sp, m)| {
                update_internal_mttest(
                    d,
                    traces,
                    y,
                    pois,
                    n_evol.view(),
                    m,
                    delta_prods,
                    dpp,
                    states,
                    sp,
                    equations,
                );
            });
    }

    pub fn get_ttest(&self) -> Array1<f64> {
        let mut ret = Array1::<f64>::zeros((self.ns,));
        let n = self.states.len();
        let n0 = self.n_samples[0];
        let n1 = self.n_samples[1];

        // find the c that will contains the variances
        let s = self.states_plain.slice(s![.., self.states[n - 1][0].1, ..]);
        let expcted: Vec<usize> = (0..self.d).collect();

        let u: Vec<&(Vec<usize>, usize)> = (&self.states[self.d - 2])
            .into_iter()
            .filter(|(c, _)| {
                izip!(c.iter(), expcted.iter())
                    .filter(|(x, y)| x == y)
                    .count()
                    == self.d
            })
            .collect();

        assert!(u.len() == 1);

        let u = self.states_plain.slice(s![.., (u[0]).1, ..]);
        let mut u0: Array1<f64> = (u.slice(s![0 as usize, ..])).to_owned();
        u0 /= n0 as f64;
        let mut s0: Array1<f64> = (s.slice(s![0 as usize, ..])).to_owned();
        s0 /= n0 as f64;

        let mut u1: Array1<f64> = (u.slice(s![1 as usize, ..])).to_owned();
        u1 /= n1 as f64;
        let mut s1: Array1<f64> = (s.slice(s![1 as usize, ..])).to_owned();
        s1 /= n1 as f64;

        ret.assign(&(&u0 - &u1));

        u0.mapv_inplace(|x| x.powi(2));
        s0 -= &u0;
        s0 /= n0 as f64;
        u1.mapv_inplace(|x| x.powi(2));
        s1 -= &u1;
        s1 /= n1 as f64;

        let mut den = &s1 + &s0;
        den.mapv_inplace(|x| f64::sqrt(x));

        ret /= &den;

        ret
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

fn update_internal_mttest(
    d: usize,
    traces: ArrayView2<i16>,
    y: ArrayView1<u16>,
    pois: ArrayView2<u64>,

    n_evol: ArrayView1<f64>,
    mut m: ArrayViewMut3<f64>,
    delta_prods: &Vec<Vec<(Vec<usize>, usize, usize, usize)>>,
    mut delta_prods_plain: ArrayViewMut2<f64>,

    states: &Vec<Vec<(Vec<usize>, usize)>>,
    mut states_plain: ArrayViewMut3<f64>,
    equations: &Vec<Vec<(usize, usize, Vec<(usize, usize, usize, usize, f64)>)>>,
) {
    izip!(traces.outer_iter(), y.iter(), n_evol.iter()).for_each(|(t, y, n)| {
        // update the first mean estimates
        izip!(
            pois.outer_iter(),
            m.outer_iter_mut(),
            (delta_prods[0]).iter()
        )
        .for_each(|(poi, mut m, delta)| {
            let m = m.slice_mut(s![*y as usize, ..]);
            let m = m.into_slice().unwrap();
            //let d = delta.1.as_slice_mut().unwrap();
            let mut tmp = delta_prods_plain.slice_mut(s![delta.1, ..]);
            let d = tmp.as_slice_mut().unwrap();
            let poi = poi.as_slice().unwrap();
            let t = t.as_slice().unwrap();
            gen_delta(m, d, t, poi, *n as f64);
        });

        // update the delta_prods
        for size in 2..(2 * d + 1) {
            // size of the delta to update
            let (_, to_updates) = delta_prods.split_at(size - 1);
            let mid_index = to_updates[0][0].1;
            let (as_input_plain, mut to_updates_plain) =
                delta_prods_plain.view_mut().split_at(Axis(0), mid_index);

            to_updates[0].iter().for_each(|to_update| {
                let low_data = to_update.2;
                let up_data = to_update.3;

                let mut tmp_prod = to_updates_plain.slice_mut(s![to_update.1 - mid_index, ..]);
                let prod = tmp_prod.as_slice_mut().unwrap();

                let tmp_low_data = as_input_plain.slice(s![low_data, ..]);
                let low_data = tmp_low_data.as_slice().unwrap();

                let tmp_up_data = as_input_plain.slice(s![up_data, ..]);
                let up_data = tmp_up_data.as_slice().unwrap();

                prod_update_single(prod, low_data, up_data);
            });
        }

        izip!(2..(2 * d + 1)).rev().for_each(|eq_size| {
            let eq = &(equations[eq_size - 2]);
            let (as_input, to_updates) = states.split_at(eq_size - 2);
            let delta_prods = delta_prods;
            let mid_index = to_updates[0][0].1;
            let mut tmp = states_plain.slice_mut(s![*y as usize, .., ..]);
            let (states_asinput, mut states_to_update) =
                tmp.view_mut().split_at(Axis(0), mid_index);

            izip!(to_updates[0].iter(), eq.iter()).for_each(
                |(to_update, (full_size, full_id, subs))| {
                    let vec = states_to_update
                        .slice_mut(s![to_update.1 - mid_index, ..])
                        .into_slice()
                        .unwrap();
                    let size = eq_size;
                    let cst = ((-1.0_f64).powi(size as i32) * (*n as f64 - 1.0)
                        + ((*n as f64 - 1.0).powi(size as i32)))
                        / (*n as f64).powi(size as i32);

                    let id = delta_prods[*full_size][*full_id].1;
                    let tmp = delta_prods_plain.slice(s![id, ..]);
                    let d = tmp.as_slice().unwrap();
                    prod_update_add(vec, d, cst);

                    subs.iter().for_each(|(p0, p1, d0, d1, cst)| {
                        let p = states_asinput.slice(s![as_input[*p0][*p1].1, ..]);

                        let cst = cst * (1.0 / (*n as f64)).powi(*d0 as i32 + 1);

                        let pv = p.as_slice().unwrap();
                        let id = delta_prods[*d0][*d1].1;
                        let tmp = delta_prods_plain.slice(s![id, ..]);
                        let d = tmp.as_slice().unwrap();
                        prod_update(vec, pv, d, cst);
                    });
                },
            );
        });
    });
}
