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
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_integer::binomial;
use rayon::prelude::*;

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
    /// Number of samples per trace
    states: Vec<Vec<(Vec<usize>, Array2<f64>)>>,
    /// POIS
    pois: Array2<u64>,
    /// Central first of moments
    m: Array3<f64>,
    d: usize,
    ns: usize,
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
        let states: Vec<Vec<(Vec<usize>, Array2<f64>)>> = (2..(2 * d + 1))
            .map(|l| {
                let mut tmp: Vec<Vec<usize>> = sets.clone().into_iter().combinations(l).collect();

                tmp.iter_mut().for_each(|x| x.sort());
                let combi_single: Vec<Vec<usize>> = tmp.clone().into_iter().unique().collect();
                /*                let count: Vec<usize> = combi_single
                                    .iter()
                                    .map(|combi| tmp.iter().filter(|x| *x == combi).count())
                                    .collect();
                */
                combi_single
                    .into_iter()
                    .map(|x| (x, Array2::<f64>::zeros((2, ns))))
                    .collect()
            })
            .collect();

        //println!("{:#?}", states);
        MTtest {
            n_samples: Array1::<u64>::zeros((2,)),
            pois: pois.to_owned(),
            states: states,
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

        let mut delta = Array2::<f64>::zeros((self.d, self.ns));
        izip!(traces.outer_iter(), y.iter(), n_evol.iter()).for_each(|(t, y, n)| {
            let d = self.d;
            let pois = &self.pois;
            let m = &mut self.m;
            let ns = self.ns;

            // update the first mean estimates
            izip!(
                pois.outer_iter(),
                m.outer_iter_mut(),
                delta.outer_iter_mut()
            )
            .for_each(|(poi, mut m, mut delta)| {
                let ordered_t = poi.mapv(|x| t[x as usize] as f64);
                let mut m = m.slice_mut(s![*y as usize, ..]);
                delta.assign(&(&ordered_t - &m));
                m += &(&delta / (*n));
            });

            for size in (2..(2 * d + 1)).rev() {
         //       println!("\n Up sets of size {} input {}", size, y);

                // split between was will be used to update and what will update
                let (as_input, to_updates) = self.states.split_at_mut(size - 2);

                let to_updates = &mut to_updates[0];

                // all the combinations to update with this size
                to_updates.iter_mut().for_each(|(combi, vec)| {
                    assert!(combi.len() == size);

                    let vec = &mut vec.slice_mut(s![*y as usize, ..]);
                    let mut acc_vec = Array1::<f64>::zeros((ns,));

                    // update with all the sub Cs
                    for l in 2..size {
                        // the combinations with current size l
                        let as_i = &as_input[l - 2];
                        // all combinations to update the current set
                        let mut tmp: Vec<Vec<usize>> =
                            combi.clone().into_iter().combinations(l).collect();

                        // get the unique ones and count their occurence
                        tmp.iter_mut().for_each(|x| x.sort());
                        let combi_single: Vec<Vec<usize>> =
                            tmp.clone().into_iter().unique().collect();
                        let count: Vec<usize> = combi_single
                            .iter()
                            .map(|combi| tmp.iter().filter(|x| *x == combi).count())
                            .collect();

                        // the associated c vector
                        let c_vecs: Vec<&Array2<f64>> = combi_single
                            .iter()
                            .map(|c| {
                                &(as_i
                                    .into_iter()
                                    .filter(|as_i| *as_i.0 == *c)
                                    .collect::<Vec<&(Vec<usize>, Array2<f64>)>>()[0]
                                    .1)
                            })
                            .collect();

                        izip!(count.iter(), combi_single.iter(), c_vecs.iter()).for_each(
                            |(count, current_combi, c_vec)| {
                                let c_vec = &c_vec.slice(s![*y as usize, ..]);
                                acc_vec.assign(c_vec);
                                acc_vec *= *count as f64;

                                // compute the missing ones to multiply the deltas
                                let mut missing = combi.clone();
                                current_combi.iter().for_each(|c| {
                                    let posi = missing.iter().position(|x| *x == *c).unwrap();
                                    missing.remove(posi);
                                });

                                missing.iter().for_each(|c| {
                                    acc_vec /= -1.0 * *n as f64;
                                    let x = delta.slice(s![*c as usize, ..]);
                                    acc_vec *= &x;
                                });

                                *vec += &acc_vec;
                            },
                        );
                    }

                    // add the last terms with product of delta
                    acc_vec.fill(
                        ((-1.0_f64).powi(size as i32) * (*n as f64 - 1.0)
                            + ((*n as f64 - 1.0).powi(size as i32)))
                            / (*n as f64).powi(size as i32),
                    );

                    combi.iter().for_each(|c| {
                        acc_vec *= &delta.slice(s![*c as usize, ..]);
                    });
                    *vec += &acc_vec;
                    //println!("delta {}", delta);
                });
            }
        });
        //println!("first order {:#?}", self.m);
        //println!("state {:#?}", self.states);
    }

    pub fn get_ttest(&self) -> Array1<f64> {
        let mut ret = Array1::<f64>::zeros((self.ns,));
        let n = self.states.len();
        let n0 = self.n_samples[0];
        let n1 = self.n_samples[1];

        // find the c that will contains the variances
        let s = &self.states[n - 1][0].1;
        let expcted: Vec<usize> = (0..self.d).collect();

        let u: Vec<&(Vec<usize>, Array2<f64>)> = (&self.states[self.d - 2])
            .into_iter()
            .filter(|(c, _)| {
                izip!(c.iter(), expcted.iter())
                    .filter(|(x, y)| x == y)
                    .count()
                    == self.d
            })
            .collect();

        assert!(u.len() == 1);
        //println!("u {:#?}", u);

        let u = &((u[0]).1);
        let mut u0: Array1<f64> = (u.slice(s![0 as usize, ..])).to_owned();
        u0 /= n0 as f64;
        let mut s0: Array1<f64> = (s.slice(s![0 as usize, ..])).to_owned();
        s0 /= n0 as f64;

        let mut u1: Array1<f64> = (u.slice(s![1 as usize, ..])).to_owned();
        u1 /= n1 as f64;
        let mut s1: Array1<f64> = (s.slice(s![1 as usize, ..])).to_owned();
        s1 /= n1 as f64;

        //println!("u0 {}", u0);
        //println!("u1 {}", u1);
        ret.assign(&(&u0 - &u1));

        u0.mapv_inplace(|x| x.powi(2));
        s0 -= &u0;
        s0 /= n0 as f64;
        u1.mapv_inplace(|x| x.powi(2));
        s1 -= &u1;
        s1 /= n1 as f64;

        let mut den = &s1 + &s0;
        //println!("s0 {}", s0);
        //println!("s1 {}", s1);
        den.mapv_inplace(|x| f64::sqrt(x));

        ret /= &den;

        ret
    }
}
