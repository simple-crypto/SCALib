//! Estimation for Multivariate T-test.
//!
//! An estimation of MTtest is represented with a MTtest struct. Calling update allows to update
//! the MTtest state with fresh measurements. get_ttest returns the current value of the estimate.
//!
//! MultivarCSAcc computes the central sums CS. MTtest internally uses MultivarCSAcc
//! to derive statistical moment estimations.
//!
//! This is based on the one-pass algorithm proposed in
//! <https://eprint.iacr.org/2015/207> section 5 as well as <https://doi.org/10.2172/1028931>.
use itertools::{izip, Itertools};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;

// Length of chunck for traces
const NS_BATCH: usize = 1 << 8;

// Aligned f64 on 256 bits to fit AVX2 instructions.
#[derive(Clone, Debug, Copy)]
#[repr(align(32))]
struct Af64 {
    x: [f64; 4],
}

pub struct MultivarCSAcc {
    /// Number of tuples to evaluate.
    pub ns: usize,
    /// pois matrix of size (d,ns)
    pub pois: Array2<u32>,
    /// Number of classes
    pub nc: usize,
    /// Number of variables in the statistical test (and therefore, its order).
    pub d: usize,
    /// Number of samples in each sets. shape (nc,)
    pub n_traces: Array1<u64>,
    /// Estimation of all needed centered statistical moments (each multiplied by the number of samples), excluding first-order moments (which are stored in `means`).
    /// shape of (nc, combis.len(), d)
    pub cs: Array3<f64>,
    /// Current means estimates
    /// shape of (nc, d, ns)
    pub mean: Array3<f64>,
    /// List of all the variable combinations to compute the centered moment of.
    /// This contains all the unique combinations on the set (0..d) U (0..d).
    /// The combinations are ordered by size.
    pub combis: Vec<Vec<usize>>,
    /// Each element in `posi` describes how to compute the corresponding moment in `combis` (`posi.len() == combis.len()`). Each incremental contribution to a moment can be computed as the product between a contribution to a lower-order moment (`posi[i].1`) and a centered data point (`posi[i].0` is the index of the corresponding variable).
    pub posi: Vec<(i32, i32)>,
}

impl MultivarCSAcc {
    /// Creates an MultivarCSAcc.
    /// It allows to estimate for a class c the central sums (CS) such that
    /// at index j such that for the set combi:
    ///
    /// cs_Q[c,index(combi),j] = sum_i(prod_{s \in combi} Q[c,pois[s,j]])
    ///
    /// for which:
    ///    - combi: is a unique combination in the multiset (0..d)U(0..d). s are the elements in S.
    ///    - i: ranges over all the traces in the class c
    ///    - Q: are the centered traces
    ///    - pois: see below.
    ///    - j: Index of the cenral sum to compute.
    ///
    /// The update of CS_Q estimation with fresh measurements Q' is done in two steps:
    ///     1. Compute the CS_Q' of Q' with a two-passes algorithm
    ///     2. Merge CS_Q and CS_Q' to obtain CS_{Q U Q'}. This is performed
    ///     thanks to the algorithms in <https://doi.org/10.2172/1028931> and
    ///     <https://eprint.iacr.org/2015/207>
    ///
    /// pois : (d,ns) array where each line corresponds to a poi
    /// nc : Number of classes to estimate the cs
    pub fn new(pois: ArrayView2<u32>, nc: usize) -> Self {
        let ns = pois.shape()[1];
        let d = pois.shape()[0];
        let max_set: Vec<usize> = (0..d).chain(0..d).collect();

        // generate all the unique subsets of (0..d, 0..d)
        // each of them will maintain a state to be updated
        let combis: Vec<Vec<usize>> = if d > 2 {
            (1..=(2 * d))
                .map(|l| {
                    max_set
                        .clone()
                        .into_iter()
                        .combinations(l)
                        .map(|mut x| {
                            x.sort();
                            x
                        })
                        .unique()
                })
                .flatten()
                .collect()
        } else {
            // This is hard-coded to ensure that the order matches our hard-coded algorithm for `d=2`.
            vec![
                vec![0],
                vec![1],
                vec![0, 0],
                vec![0, 1],
                vec![1, 1],
                vec![0, 0, 1],
                vec![0, 1, 1],
                vec![0, 0, 1, 1],
            ]
        };

        let posi = combis
            .clone()
            .iter_mut()
            .map(|combi| {
                if combi.len() == 1 {
                    (combi[0] as i32, -1)
                } else {
                    let c = combi.remove(0);
                    (
                        c as i32,
                        combis.iter().position(|x| x == combi).unwrap() as i32,
                    )
                }
            })
            .collect();

        MultivarCSAcc {
            ns: ns,
            pois: pois.to_owned(),
            nc: nc,
            d: d,
            n_traces: Array1::<u64>::zeros((nc,)),
            cs: Array3::<f64>::zeros((nc, combis.len(), ns)),
            mean: Array3::<f64>::zeros((nc, d, ns)),
            combis: combis,
            posi: posi,
        }
    }

    /// Merges the current CS estimate another CS estimate.
    ///
    /// cs_q2 : A CS estimate to merge with the current estimation. Its shape is
    /// (nc,combis.len(),ns)
    /// u_q2 : mean per class in the other estimation (nc,d,ns)
    /// n_traces_q2 : Number of traces in each of the classes (nc,)

    // We next describe the merge rule for a single class.
    //
    // Definitions:
    //  - Q1 and Q2 are the set of traces used to build the two CS's to merge.
    //  - Q = Q1 U Q2
    //  - n = |Q|
    //  - 0 <= s < d
    //  - u[s,j] = mean(Q[:,pois[s,j]])
    //  - delta[s,j] = u2[s,j] - u1[s,j]
    //
    //  Update rule:
    //  cs_q[combi,j] = cs_q1[combi,j] + cs_q2[combi,j] + (1) + (2)
    //  with:
    //
    //  (1) = sum_{k=2}^{|combi|-1}(
    //
    //          sum_{S in |combi| | |S| = k} cs_q1[S,j] prod_{i in (|combi|\S)} (-n_1 / n) *
    //                                      delta[i,j]
    //          +
    //          ((-n2 / n ) ** |combi|) n1 prod_{j in combi) delta[j]
    //        )
    //  (2) is the symmetry of (1) by swapping q1 and q2.
    pub fn merge_from_state(
        &mut self,
        cs2: ArrayView3<f64>,
        u2: ArrayView3<f64>,
        n2: ArrayView1<u64>,
    ) {
        // each row will contain one power of deltas
        let combis = &self.combis;
        let ns = self.ns;
        izip!(
            self.cs.outer_iter_mut(),
            self.mean.outer_iter_mut(),
            self.n_traces.iter(),
            cs2.outer_iter(),
            u2.outer_iter(),
            n2.iter()
        )
        .for_each(|(mut cs1, mut u1, n1, cs2, u2, n2)| {
            let n1 = *n1 as f64;
            let n2 = *n2 as f64;
            let n = n1 + n2;
            if n1 == 0.0 {
                cs1.assign(&cs2);
                u1.assign(&u2);
            } else if n2 != 0.0 {
                let delta = &u2 - &u1;
                let mut prod1 = Array1::<f64>::ones((ns,));
                let mut prod2 = Array1::<f64>::ones((ns,));

                // update all the combinations one by one.
                for (i, combi) in combis.iter().enumerate().rev() {
                    // split cs between the inputs and the outputs
                    let (cs1_smaller, mut cs1_larger) = cs1.view_mut().split_at(Axis(0), i);
                    let (cs2_smaller, cs2_larger) = cs2.view().split_at(Axis(0), i);
                    let mut cs1_larger = cs1_larger.slice_mut(s![0, ..]);

                    cs1_larger += &cs2_larger.slice(s![0, ..]);

                    for k in 2..combi.len() {
                        for set in combi.iter().combinations(k) {
                            let set: Vec<usize> = set.into_iter().map(|x| *x).collect();
                            let id = combis.iter().position(|x| *x == set).unwrap();

                            let mut to_multiply: Vec<usize> =
                                combi.into_iter().map(|x| *x).collect();
                            for x in set {
                                to_multiply
                                    .remove(to_multiply.iter().position(|y| x == *y).unwrap());
                            }

                            prod1.assign(&cs1_smaller.slice(s![id, ..]));
                            prod2.assign(&cs2_smaller.slice(s![id, ..]));
                            for i in to_multiply {
                                prod1 *= &(&delta.slice(s![i, ..]) * (-n2 / n));
                                prod2 *= &(&delta.slice(s![i, ..]) * (n1 / n));
                            }
                            cs1_larger += &prod1;
                            cs1_larger += &prod2;
                        }
                    }

                    prod1.fill((-n2 / n).powi(combi.len() as i32) * n1);
                    prod2.fill((n1 / n).powi(combi.len() as i32) * n2);
                    for x in combi.into_iter() {
                        prod1 *= &delta.slice(s![*x, ..]);
                        prod2 *= &delta.slice(s![*x, ..]);
                    }
                    cs1_larger += &prod1;
                    cs1_larger += &prod2;
                    if combi.len() == 1 {
                        cs1_larger.fill(0.0);
                    }
                }
                u1 += &((&u2 - &u1) * (n2 / n));
            }
        });
        self.n_traces += &n2;
    }

    /// Merges to different CS struct estimations.
    pub fn merge(&mut self, other: &Self) {
        self.merge_from_state(other.cs.view(), other.mean.view(), other.n_traces.view());
    }

    /// Updates the current CS estimation with fresh traces and its means per class
    /// This workds in two steps:
    ///     1. Computes CS on the fresh traces.
    ///     2. Merge this with the current CS.
    ///
    /// t0 : fresh traces for set 0. of shape (ns, ceil(n0 // 4)).
    /// t1 : fresh traces for set 1. of shape (ns, ceil(n1 // 4)).
    /// mean : mean per class of the all traces
    /// n_traces : count per classes in traces
    fn update_with_means(
        &mut self,
        t0: ArrayView2<Af64>,
        t1: ArrayView2<Af64>,
        mean: ArrayView2<f64>,
        n_traces: ArrayView1<u64>,
    ) {
        // Computes CS on the traces with a 2 passes algorithm.
        let mut cs2 = Array3::<f64>::zeros((2, self.combis.len(), self.pois.shape()[1]));
        izip!(self.pois.axis_iter(Axis(1)), cs2.axis_iter_mut(Axis(2))).for_each(
            |(pois, mut cs2)| {
                if self.d == 2 {
                    let mut acc00 = Af64 { x: [0.0; 4] };
                    let mut acc01 = Af64 { x: [0.0; 4] };
                    let mut acc11 = Af64 { x: [0.0; 4] };
                    let mut acc001 = Af64 { x: [0.0; 4] };
                    let mut acc011 = Af64 { x: [0.0; 4] };
                    let mut acc0011 = Af64 { x: [0.0; 4] };
                    inner_prod_d2(
                        &mut acc00,
                        &mut acc01,
                        &mut acc11,
                        &mut acc001,
                        &mut acc011,
                        &mut acc0011,
                        t0.slice(s![pois[0] as usize, ..]).to_slice().unwrap(),
                        t0.slice(s![pois[1] as usize, ..]).to_slice().unwrap(),
                    );
                    for i in 0..4 {
                        cs2[[0, 2 + 0]] += acc00.x[i];
                        cs2[[0, 2 + 1]] += acc01.x[i];
                        cs2[[0, 2 + 2]] += acc11.x[i];
                        cs2[[0, 2 + 3]] += acc001.x[i];
                        cs2[[0, 2 + 4]] += acc011.x[i];
                        cs2[[0, 2 + 5]] += acc0011.x[i];
                    }
                } else {
                    let mut accs = vec![Af64 { x: [0.0; 4] }; self.combis.len()];
                    let mut prods = vec![Af64 { x: [0.0; 4] }; self.combis.len()];
                    let ts = pois
                        .iter()
                        .map(|x| t0.slice(s![*x as usize, ..]).to_slice().unwrap())
                        .collect();

                    inner_prod_generic(
                        accs.as_mut_slice(),
                        prods.as_mut_slice(),
                        self.posi.as_slice(),
                        &ts,
                    );
                    for i in 0..4 {
                        for j in 0..self.combis.len() {
                            cs2[[0, j]] += accs[j].x[i];
                        }
                    }
                }

                if self.d == 2 {
                    let mut acc00 = Af64 { x: [0.0; 4] };
                    let mut acc01 = Af64 { x: [0.0; 4] };
                    let mut acc11 = Af64 { x: [0.0; 4] };
                    let mut acc001 = Af64 { x: [0.0; 4] };
                    let mut acc011 = Af64 { x: [0.0; 4] };
                    let mut acc0011 = Af64 { x: [0.0; 4] };
                    inner_prod_d2(
                        &mut acc00,
                        &mut acc01,
                        &mut acc11,
                        &mut acc001,
                        &mut acc011,
                        &mut acc0011,
                        t1.slice(s![pois[0] as usize, ..]).to_slice().unwrap(),
                        t1.slice(s![pois[1] as usize, ..]).to_slice().unwrap(),
                    );
                    for i in 0..4 {
                        cs2[[1, 2 + 0]] += acc00.x[i];
                        cs2[[1, 2 + 1]] += acc01.x[i];
                        cs2[[1, 2 + 2]] += acc11.x[i];
                        cs2[[1, 2 + 3]] += acc001.x[i];
                        cs2[[1, 2 + 4]] += acc011.x[i];
                        cs2[[1, 2 + 5]] += acc0011.x[i];
                    }
                } else {
                    let mut prods = vec![Af64 { x: [0.0; 4] }; self.combis.len()];
                    let mut accs = vec![Af64 { x: [0.0; 4] }; self.combis.len()];
                    let ts = pois
                        .iter()
                        .map(|x| t1.slice(s![*x as usize, ..]).to_slice().unwrap())
                        .collect();

                    inner_prod_generic(
                        accs.as_mut_slice(),
                        prods.as_mut_slice(),
                        self.posi.as_slice(),
                        &ts,
                    );
                    for i in 0..4 {
                        for j in 0..self.combis.len() {
                            cs2[[1, j]] += accs[j].x[i];
                        }
                    }
                }
            },
        );

        // Genereted the means according to the pois
        let pois = &self.pois;
        let mut mapped_means = Array3::<f64>::zeros((self.nc, self.d, self.ns));
        izip!(mapped_means.outer_iter_mut(), mean.outer_iter()).for_each(
            |(mut to_update, mean)| {
                izip!(to_update.outer_iter_mut(), pois.outer_iter()).for_each(
                    |(mut to_update, pois)| {
                        to_update.zip_mut_with(&pois, |x, p| *x = mean[*p as usize]);
                    },
                );
            },
        );

        // Merge CS of the inputs with self.
        self.merge_from_state(cs2.view(), mapped_means.view(), n_traces.view());
    }

    /// Updates the current CS estimation with fresh traces
    /// traces : fresh traces
    /// y : class corresponding to each traces
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let (mean, n_traces) = means_per_class(traces, y, self.nc);
        let (t0, t1) = center_transpose_align(traces, mean.view(), y);
        self.update_with_means(t0.view(), t1.view(), mean.view(), n_traces.view());
    }
}

pub struct MTtest {
    /// order of the test
    d: usize,
    /// Number of samples per trace
    ns: usize,
    /// Vector of CS accumulators. This will be used to
    /// derive the statistical moments.
    accumulators: Vec<MultivarCSAcc>,
    /// Pois to combine in the multivariate T-test (d,ns)
    pois: Array2<u32>,
}

impl MTtest {
    /// Create a new Ttest state.
    /// ns: traces length
    /// d: order of the Ttest
    pub fn new(d: usize, pois: ArrayView2<u32>) -> Self {
        assert!(d == pois.shape()[0]);
        assert!(
            d > 1,
            "Order of Multivariate T-test should be larger than 1, provided d = {}",
            d
        );

        // generates the MultivarCSAcc underlying MTtest.
        let accumulators = build_accumulator(pois.view());
        MTtest {
            d: d,
            ns: pois.shape()[1],
            accumulators: accumulators,
            pois: pois.to_owned(),
        }
    }

    /// Update the MTtest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        // First pass to compute the means, center and align.
        let (mean, n_traces) = means_per_class(traces, y, 2);
        let (t0, t1) = center_transpose_align(traces, mean.view(), y);

        // chunck different traces for more threads
        (
            self.pois.axis_chunks_iter(Axis(1), NS_BATCH),
            &mut self.accumulators,
        )
            .into_par_iter()
            .for_each(|(_, acc)| {
                // chunck the traces with their lenght
                acc.update_with_means(t0.view(), t1.view(), mean.view(), n_traces.view());
            });
    }

    /// Computes the t statistic according to the
    /// order d.
    pub fn get_ttest(&self) -> Array1<f64> {
        let mut t = Array1::<f64>::zeros((self.ns,));
        izip!(
            t.axis_chunks_iter_mut(Axis(0), NS_BATCH),
            self.accumulators.iter()
        )
        .for_each(|(mut t, acc)| {
            t.fill(1.0);
            let n1 = acc.n_traces[0] as f64;
            let n2 = acc.n_traces[1] as f64;

            let mut mu1 = Array1::<f64>::zeros((t.shape()[0],));
            let mut var1 = Array1::<f64>::zeros((t.shape()[0],));
            let mut mu2 = Array1::<f64>::zeros((t.shape()[0],));
            let mut var2 = Array1::<f64>::zeros((t.shape()[0],));

            // assign means
            let combi: Vec<usize> = (0..self.d).collect();
            let id = acc.combis.iter().position(|x| *x == combi).unwrap();
            let mus = acc.cs.slice(s![.., id, ..]);
            mu1.assign(&(&mus.slice(s![0, ..]) / n1));
            mu2.assign(&(&mus.slice(s![1, ..]) / n2));

            let mut combi: Vec<usize> = (0..self.d).chain(0..self.d).collect();
            combi.sort();
            let id = acc.combis.iter().position(|x| *x == combi).unwrap();
            let vars = acc.cs.slice(s![.., id, ..]);
            var1.assign(&(&vars.slice(s![0, ..]) / n1));
            var2.assign(&(&vars.slice(s![1, ..]) / n2));

            if self.d > 2 {
                for j in 0..self.d {
                    let combi: Vec<usize> = vec![j, j];
                    let id = acc.combis.iter().position(|x| *x == combi).unwrap();
                    let mus = acc.cs.slice(s![.., id, ..]);
                    mu1 /= &(mus.slice(s![0, ..]).mapv(|x| (x / n1).sqrt()));
                    mu2 /= &(mus.slice(s![1, ..]).mapv(|x| (x / n2).sqrt()));

                    var1 /= &(mus.slice(s![0, ..]).mapv(|x| (x / n1)));
                    var2 /= &(mus.slice(s![1, ..]).mapv(|x| (x / n2)));
                }
            }
            var1 -= &(&mu1 * &mu1);
            var2 -= &(&mu2 * &mu2);

            t.assign(&(&mu1 - &mu2));

            t /= &((&var1 / n1) + (&var2 / n2)).mapv(|x| x.sqrt());
        });
        t
    }
}

/// Computes the means per class
fn means_per_class(
    traces: ArrayView2<i16>,
    y: ArrayView1<u16>,
    nc: usize,
) -> (Array2<f64>, Array1<u64>) {
    let mut sum = Array2::<i32>::zeros((nc, traces.shape()[1]));
    let mut sum64 = Array2::<i64>::zeros((nc, traces.shape()[1]));
    let mut n_traces = Array1::<u64>::zeros(nc);

    for (trace, class) in traces.outer_iter().zip(y.iter()) {
        n_traces[*class as usize] += 1;
        let mut s = sum.slice_mut(s![*class as usize, ..]);
        if (n_traces[*class as usize] % (1 << 16)) == ((1 << 16) - 1) {
            let mut s64 = sum64.slice_mut(s![*class as usize, ..]);
            s64 += &s.mapv(|x| x as i64);
            s.fill(0);
        }
        s.zip_mut_with(&trace, |s, t| {
            *s += *t as i32;
        });
    }
    sum64 += &sum.mapv(|x| x as i64);
    let n = n_traces.mapv(|x| x as f64);
    let mut mean = sum64.mapv(|x| x as f64);
    mean.axis_iter_mut(Axis(1)).for_each(|mut m| m /= &n);

    (mean, n_traces)
}

/// Generates accumulators according to the pois.
fn build_accumulator(pois: ArrayView2<u32>) -> Vec<MultivarCSAcc> {
    // number of required accumulators
    let ns = pois.shape()[1];
    let n_batches = ((ns as f64) / (NS_BATCH as f64)).ceil() as usize;
    let accumulators: Vec<MultivarCSAcc> = (0..n_batches)
        .map(|x| {
            let l = std::cmp::min(ns - (x * NS_BATCH), NS_BATCH);
            MultivarCSAcc::new(pois.slice(s![.., (NS_BATCH * x)..(NS_BATCH * x + l)]), 2)
        })
        .collect();

    accumulators
}

#[inline(never)]
fn inner_prod_generic(
    accs: &mut [Af64],
    prods: &mut [Af64],
    posi: &[(i32, i32)],
    ts: &Vec<&[Af64]>,
) {
    for j in 0..ts[0].len() {
        for i in 0..prods.len() {
            let (t, c) = posi[i];
            let (prod_old, mut prod_up) = prods.split_at_mut(i);
            let (_, mut accs_up) = accs.split_at_mut(i);

            let t = &ts[t as usize][j];
            if c == -1 {
                prod_up[0].x[0] = t.x[0];
                prod_up[0].x[1] = t.x[1];
                prod_up[0].x[2] = t.x[2];
                prod_up[0].x[3] = t.x[3];
            } else {
                prod_up[0].x[0] = prod_old[c as usize].x[0] * t.x[0];
                prod_up[0].x[1] = prod_old[c as usize].x[1] * t.x[1];
                prod_up[0].x[2] = prod_old[c as usize].x[2] * t.x[2];
                prod_up[0].x[3] = prod_old[c as usize].x[3] * t.x[3];
            }

            accs_up[0].x[0] += prod_up[0].x[0];
            accs_up[0].x[1] += prod_up[0].x[1];
            accs_up[0].x[2] += prod_up[0].x[2];
            accs_up[0].x[3] += prod_up[0].x[3];
        }
    }
}

#[inline(never)]
fn inner_prod_d2(
    acc00: &mut Af64,
    acc01: &mut Af64,
    acc11: &mut Af64,
    acc001: &mut Af64,
    acc011: &mut Af64,
    acc0011: &mut Af64,

    t0: &[Af64],
    t1: &[Af64],
) {
    izip!(t0.iter(), t1.iter()).for_each(|(t0, t1)| {
        acc00.x[0] += t0.x[0] * t0.x[0];
        acc00.x[1] += t0.x[1] * t0.x[1];
        acc00.x[2] += t0.x[2] * t0.x[2];
        acc00.x[3] += t0.x[3] * t0.x[3];

        acc01.x[0] += t0.x[0] * t1.x[0];
        acc01.x[1] += t0.x[1] * t1.x[1];
        acc01.x[2] += t0.x[2] * t1.x[2];
        acc01.x[3] += t0.x[3] * t1.x[3];

        acc11.x[0] += t1.x[0] * t1.x[0];
        acc11.x[1] += t1.x[1] * t1.x[1];
        acc11.x[2] += t1.x[2] * t1.x[2];
        acc11.x[3] += t1.x[3] * t1.x[3];

        acc001.x[0] += t0.x[0] * t0.x[0] * t1.x[0];
        acc001.x[1] += t0.x[1] * t0.x[1] * t1.x[1];
        acc001.x[2] += t0.x[2] * t0.x[2] * t1.x[2];
        acc001.x[3] += t0.x[3] * t0.x[3] * t1.x[3];

        acc011.x[0] += t0.x[0] * t1.x[0] * t1.x[0];
        acc011.x[1] += t0.x[1] * t1.x[1] * t1.x[1];
        acc011.x[2] += t0.x[2] * t1.x[2] * t1.x[2];
        acc011.x[3] += t0.x[3] * t1.x[3] * t1.x[3];

        acc0011.x[0] += t0.x[0] * t0.x[0] * t1.x[0] * t1.x[0];
        acc0011.x[1] += t0.x[1] * t0.x[1] * t1.x[1] * t1.x[1];
        acc0011.x[2] += t0.x[2] * t0.x[2] * t1.x[2] * t1.x[2];
        acc0011.x[3] += t0.x[3] * t0.x[3] * t1.x[3] * t1.x[3];
    });
}

fn center_transpose_align(
    traces: ArrayView2<i16>,
    means: ArrayView2<f64>,
    y: ArrayView1<u16>,
) -> (Array2<Af64>, Array2<Af64>) {
    let ns = traces.shape()[1];

    let posi0: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, x)| **x == 0)
        .map(|(i, _)| i as usize)
        .collect();
    let n0 = posi0.len();

    let mut t0 = Array2::<Af64>::from_elem(
        (ns, (n0 as f64 / 4.0).ceil() as usize),
        Af64 {
            x: [0.0, 0.0, 0.0, 0.0],
        },
    );
    let posi1: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, x)| **x == 1)
        .map(|(i, _)| i as usize)
        .collect();
    let n1 = posi1.len();

    let mut t1 = Array2::<Af64>::from_elem(
        (ns, (n1 as f64 / 4.0).ceil() as usize),
        Af64 {
            x: [0.0, 0.0, 0.0, 0.0],
        },
    );

    (traces.axis_iter(Axis(1)), t0.axis_iter_mut(Axis(0)))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (traces, mut t0))| {
            let m0 = means[[0, i]];
            izip!(&posi0.iter().chunks(4), t0.iter_mut()).for_each(|(p, t)| {
                p.enumerate().for_each(|(x, p)| {
                    t.x[x] = traces[*p] as f64 - m0;
                });
            });
        });

    (traces.axis_iter(Axis(1)), t1.axis_iter_mut(Axis(0)))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (traces, mut t1))| {
            let m1 = means[[1, i]];
            izip!(&posi1.iter().chunks(4), t1.iter_mut()).for_each(|(p, t)| {
                p.enumerate().for_each(|(x, p)| {
                    t.x[x] = traces[*p] as f64 - m1;
                });
            });
        });

    (t0, t1)
}
