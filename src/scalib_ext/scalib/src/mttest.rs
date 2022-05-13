//! Estimation for higher-order Multivariate T-test.
//!
//! An estimation of MTtest is represented with a MTtest struct. Calling update allows
//! to update the Ttest state with fresh measurements. get_ttest returns the current value
//! of the estimate.
//!
//! This is based on the one-pass algorithm proposed in
//! <https://eprint.iacr.org/2015/207>.
use itertools::{izip, Itertools};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, Axis};
use rayon::prelude::*;
use std::cmp;
// Length of chunck for traces
const NS_BATCH: usize = 1 << 13;

pub struct MultivarCSAcc {
    /// Number of tuples to evaluate.
    pub ns: usize,
    /// pois matrix of size (d,ns)
    pub pois: Array2<u32>,
    /// Number of classes
    pub nc: usize,
    pub d: usize,
    /// Number of samples in each sets. shape (nc,)
    pub n_traces: Array1<u64>,
    /// Current estimation of centered sums at higher order.
    /// shape of (nc, combis.len(), d)
    pub cs: Array3<f64>,
    /// Current means estimates
    pub mean: Array3<f64>,
    /// List of all the points combinations to compute the centered product sum.
    /// This contains all the unique combinations on the set (0..d) U (0..d).
    /// The combinations are ordered by size.
    pub combis: Vec<Vec<usize>>,
}

impl MultivarCSAcc {
    /// Creates an MultivarCSAcc
    /// pois : (d,ns) array where each line corresponds to a poi
    /// nc : Number of classes to estimate the higher order cs
    pub fn new(pois: ArrayView2<u32>, nc: usize) -> Self {
        let ns = pois.shape()[1];
        let d = pois.shape()[0];
        let max_set: Vec<usize> = (0..d).chain(0..d).collect();

        // generate all the unique subsets of (0..d, 0..d)
        // each of them will maintain a state to be updated
        let combis: Vec<Vec<usize>> = if d > 2 {
            (1..(2 * d + 1))
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
                        .collect()
                })
                .collect::<Vec<Vec<Vec<usize>>>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
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

        MultivarCSAcc {
            ns: ns,
            pois: pois.to_owned(),
            nc: nc,
            d: d,
            n_traces: Array1::<u64>::zeros((nc,)),
            cs: Array3::<f64>::zeros((nc, combis.len(), ns)),
            mean: Array3::<f64>::zeros((nc, d, ns)),
            combis: combis,
        }
    }

    /// Merges two MutlivarCSAcc
    /// cs_other : Estimated cs to merge (nc,combis.len(),ns)
    /// means : mean per class (nc,d,ns)
    /// n_traces : Number of traces in each of the classes (nc,)
    // We next describe the merge rule for a single class.
    //
    // Definitions:
    //  Q_1 and Q_2 are the set of traces used to build the two CS's to merge.
    //  Q = Q_1 U Q_2
    //  x^j = leakage at index j
    //  n_i = |Q_i|
    //  u^j_i = n_i sum_{Q_i} x^j_i
    //  delta^j_2,1 = u^j_2 - u^j_1
    //  cs_{Q_i}_J = \sum_{Q_i} \prod_{j \in J} (x^j - u^j_i)
    //
    //  Update rule:
    //  cs_{Q}_J = cs_{Q_1}_{J} + cs_{Q_2}_{J} + (1) + (2)
    //  with
    //  (1) = sum_{k=2}^{|J|-1}
    //              \sum_{s \ in J_|k|} cs_{Q_1}_{s}
    //                  \prod_{j \in J \ s} (-n_1 / n) * delta^j_2_1
    //          + ((-n_2 / n ) ** |J|) n_1
    //                  \prod_{j \in J) delta^j_2_1
    //
    //  (2) is the symmetry of 1.
    pub fn merge_from_state(
        &mut self,
        cs_other: ArrayView3<f64>,
        means: ArrayView3<f64>,
        n_traces: ArrayView1<u64>,
    ) {
        // each row will contain one power of deltas
        let combis = &self.combis;
        let ns = self.ns;
        izip!(
            self.cs.outer_iter_mut(),
            self.mean.outer_iter_mut(),
            self.n_traces.iter(),
            cs_other.outer_iter(),
            means.outer_iter(),
            n_traces.iter()
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
        self.n_traces += &n_traces;
    }

    /// Merges to different CS estimations.
    pub fn merge(&mut self, other: &Self) {
        self.merge_from_state(other.cs.view(), other.mean.view(), other.n_traces.view());
    }

    /// Updates the current CS estimation with fresh traces and its means per class
    /// traces : fresh traces
    /// y : class corresponding to each traces
    /// mean : mean per class of the all traces
    /// n_traces : count per classes in traces
    pub fn update_with_means(
        &mut self,
        traces: ArrayView2<i16>,
        y: ArrayView1<u16>,
        mean: ArrayView2<f64>,
        n_traces: ArrayView1<u64>,
    ) {
        // Computes CS on the traces with a 2 passes algorithm.
        let cs_other = centered_products(
            traces,
            y,
            mean.view(),
            self.pois.view(),
            self.nc,
            &self.combis,
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
        self.merge_from_state(cs_other.view(), mapped_means.view(), n_traces.view());
    }

    /// Updates the current CS estimation with fresh traces
    /// traces : fresh traces
    /// y : class corresponding to each traces
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let (mean, n_traces) = means_per_class(traces, y, self.nc);
        self.update_with_means(traces, y, mean.view(), n_traces.view());
    }
}

pub struct MTtest {
    /// order of the test
    d: usize,
    /// Number of samples per trace
    ns: usize,
    /// Vector of Moment accumulators
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
        // The traces are chuncks with:
        // _____________________________
        // |    |    |    |    |    |  |
        // _____________________________
        // |    |    |    |    |    |  |
        // _____________________________
        // |    |    |    |    |    |  |
        // _____________________________
        //
        // Each chunck above is updated with an accumulator.

        let y_chunck_size_level1 = cmp::max(
            1024,
            traces.shape()[0]
                / (cmp::max(
                    1,
                    4 * rayon::current_num_threads() / self.accumulators.len(),
                )),
        );
        let level1_accs = (
            traces.axis_chunks_iter(Axis(0), y_chunck_size_level1),
            y.axis_chunks_iter(Axis(0), y_chunck_size_level1),
        )
            .into_par_iter()
            .map(|(traces, y)| {
                // chunck different traces for more threads
                let (mean, n_traces) = means_per_class(traces, y, 2);
                let mut accumulators = build_accumulator(self.pois.view());
                (
                    self.pois.axis_chunks_iter(Axis(1), NS_BATCH),
                    &mut accumulators,
                )
                    .into_par_iter()
                    .for_each(|(_, acc)| {
                        // chunck the traces with their lenght
                        acc.update_with_means(traces, y, mean.view(), n_traces.view())
                    });
                accumulators
            })
            .reduce(
                || build_accumulator(self.pois.view()),
                |mut x, y| {
                    // accumulate all to the self accumulator
                    x.iter_mut().zip(y.iter()).for_each(|(x, y)| x.merge(y));
                    x
                },
            );
        (&mut self.accumulators, level1_accs)
            .into_par_iter()
            .for_each(|(x, y)| x.merge(&y));
    }

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

/// Applies to = ti - m
fn center_trace(to: ArrayViewMut1<f64>, ti: ArrayView1<i16>, m: ArrayView1<f64>) {
    let to = to.into_slice().unwrap();
    let ti = ti.to_slice().unwrap();
    let m = m.to_slice().unwrap();
    izip!(to.iter_mut(), ti.iter(), m.iter()).for_each(|(to, ti, m)| {
        *to = *ti as f64 - *m;
    });
}

/// Applies to2 = v1 * v1; to1 += to2
fn add_prod(
    to1: ArrayViewMut1<f64>,
    to2: ArrayViewMut1<f64>,
    v1: ArrayView1<f64>,
    v2: ArrayView1<f64>,
) {
    let to1 = to1.into_slice().unwrap();
    let to2 = to2.into_slice().unwrap();
    let v1 = v1.to_slice().unwrap();
    let v2 = v2.to_slice().unwrap();
    izip!(to1.iter_mut(), to2.iter_mut(), v1.iter(), v2.iter()).for_each(|(to1, to2, v1, v2)| {
        *to2 = *v1 * *v2;
        *to1 += *to2;
    });
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

fn centered_products(
    traces: ArrayView2<i16>,
    y: ArrayView1<u16>,
    mean: ArrayView2<f64>,
    pois: ArrayView2<u32>,
    nc: usize,
    combis: &Vec<Vec<usize>>,
) -> Array3<f64> {
    let ns = pois.shape()[1];
    let d = pois.shape()[0];
    let mut cs_other = Array3::<f64>::zeros((nc, combis.len(), ns));

    if d > 2 {
        let mut prod = Array2::<f64>::zeros((combis.len(), ns));

        let precomp_loc: Vec<usize> = combis
            .iter()
            .filter(|x| x.len() > 1)
            .map(|x| {
                if x.len() == 2 {
                    0
                } else {
                    let mut tmp = x.clone();
                    tmp.remove(0);
                    combis.iter().position(|y| tmp == *y).unwrap() - d
                }
            })
            .collect();

        // for each trace:
        //  1. center it (t - mu)
        //  2. re-order the traces for each of the pois
        let mut ct = Array1::<f64>::zeros((traces.shape()[1],));
        for c in 0..nc {
            izip!(traces.axis_iter(Axis(0)), y.iter())
                .filter(|(_, y)| **y as usize == c)
                .for_each(|(t, y)| {
                    // prod according to poi for first (t-mu)
                    center_trace(ct.view_mut(), t.view(), mean.slice(s![*y as usize, ..]));
                    let mut to_update = cs_other.slice_mut(s![*y as usize, .., ..]);

                    // re-order according to pois.
                    for i in 0..d {
                        let mut c = prod.slice_mut(s![i, ..]);
                        let mut tmp = to_update.slice_mut(s![i, ..]);
                        c.zip_mut_with(&pois.slice(s![i, ..]), |c, p| *c = ct[*p as usize]);
                        tmp += &c;
                    }

                    // compute the product for all the higher order combinations
                    let (ct, mut higher_order) = prod.view_mut().split_at(Axis(0), d);
                    let (_, mut to_update) = to_update.view_mut().split_at(Axis(0), d);
                    let (_, higher_combi) = combis.split_at(d);
                    izip!(
                        (0..higher_order.len()),
                        higher_combi.iter(),
                        precomp_loc.iter(),
                        to_update.axis_iter_mut(Axis(0)),
                    )
                    .for_each(|(i, combi, precomp_loc, to_update)| {
                        let (tmp, mut hf) = higher_order.view_mut().split_at(Axis(0), i);
                        let h = hf.slice_mut(s![0, ..]);

                        if combi.len() == 2 {
                            add_prod(
                                to_update,
                                h,
                                ct.slice(s![combi[0], ..]),
                                ct.slice(s![combi[1], ..]),
                            );
                        } else {
                            add_prod(
                                to_update,
                                h,
                                ct.slice(s![combi[0], ..]),
                                tmp.slice(s![*precomp_loc, ..]),
                            );
                        }
                    });
                });
        }
    } else {
        let mut ct = Array1::<f64>::zeros((traces.shape()[1],));
        let nt = y.iter().filter(|x| **x as usize == 0).count();
        let mut rt = Array3::<Af64>::from_elem(( ns, 2 , nt/4),Af64{x:[0.0,0.0,0.0,0.0]});
        for c in 0..nc {

/*            izip!(traces.axis_iter(Axis(0)), y.iter())
                .filter(|(_, y)| **y as usize == c)
                .enumerate()
                .for_each(|(j, (t, y))| {
                    // prod according to poi for first (t-mu)
                    center_trace(ct.view_mut(), t.view(), mean.slice(s![*y as usize, ..]));

                    // re-order according to pois.
                    for i in 0..d {
                        let mut reorder = rt.slice_mut(s![j, i, ..]);
                        let mut tmp = cs_other.slice_mut(s![c, i, ..]);
                        reorder.zip_mut_with(&pois.slice(s![i, ..]), |c, p| *c = ct[*p as usize]);
                        tmp += &reorder;
                    }
                });
*/
            let acc = center_prod_2(rt.view());
            for i in 0..6 {
                let mut cs = cs_other.slice_mut(s![c, 2 + i, ..]);
                cs.zip_mut_with(&acc, |cs, acc| *cs = acc[i]);
            }
        }
    }
    cs_other
}

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

#[derive(Clone)]
#[repr(align(32))]
pub struct Af64 {
    x: [f64; 4],
}

#[inline(never)]
pub fn center_prod_2(t: ArrayView3<Af64>) -> Array1<[f64; 6]> {
    let ns = t.shape()[0];
    let mut acc = Array1::<[f64; 6]>::from_elem((ns,), [0.0; 6]); //
    izip!(
        t.exact_chunks((1, 2, t.shape()[2])).into_iter(),
        acc.exact_chunks_mut(1).into_iter()
    )
    .for_each(|(t, mut acc)| {
        let mut acc00 = Af64{x:[0.0;4]};
        let mut acc01 = Af64{x:[0.0;4]};
        let mut acc11 = Af64{x:[0.0;4]};
        let mut acc001 = Af64{x:[0.0;4]};
        let mut acc011 = Af64{x:[0.0;4]};
        let mut acc0011 = Af64{x:[0.0;4]};
        
        let mut x0 = Af64{x:[0.0;4]}; 
        let mut x1 = Af64{x:[0.0;4]}; 
        t.axis_iter(Axis(1)).for_each(|t| {
            x0.x[0] = t[[0,0]].x[0];
            x0.x[1] = t[[0,0]].x[1];
            x0.x[2] = t[[0,0]].x[2];
            x0.x[3] = t[[0,0]].x[3];

            x1.x[0] = t[[0,1]].x[0];
            x1.x[1] = t[[0,1]].x[1];
            x1.x[2] = t[[0,1]].x[2];
            x1.x[3] = t[[0,1]].x[3];

            inner(&mut acc00,
                &mut acc01,
                &mut acc11,
                &mut acc001,
                &mut acc011,
                &mut acc0011,
                
                &x0,
                &x1,
            );
        });
        for i in 0..4 {
            acc[0][0] += acc00.x[i];
            acc[0][1] += acc01.x[i];
            acc[0][2] += acc11.x[i];
            acc[0][3] += acc001.x[i];
            acc[0][4] += acc011.x[i];
            acc[0][5] += acc0011.x[i];
        }
    });
    acc
}

#[inline]
pub fn inner(acc00: &mut Af64,
            acc01: &mut Af64,
            acc11: &mut Af64,
            acc001: &mut Af64,
            acc011: &mut Af64,
            acc0011: &mut Af64,

    t0: &Af64,
    t1: &Af64){

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


}
