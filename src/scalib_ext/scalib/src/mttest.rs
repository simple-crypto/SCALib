use itertools::{izip, Itertools};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, Axis};
const NS_BATCH: usize = 1 << 9;

pub struct MultivarMomentAcc {
    /// Number of tuples to evaluate.
    pub ns: usize,
    /// pois matrix of size (d,ns)
    pub pois: Array2<u32>,
    /// Number of classes
    pub nc: usize,
    pub d: usize,
    /// Number of samples in each sets
    pub n_traces: Array1<u64>,
    /// Current estimation
    pub moments: Array3<f64>,
    /// Current means estimates
    pub mean: Array3<f64>,
    /// List of all the combinations
    pub combis: Vec<Vec<usize>>,
}

impl MultivarMomentAcc {
    pub fn new(pois: ArrayView2<u32>, nc: usize) -> Self {
        let ns = pois.shape()[1];
        let d = pois.shape()[0];
        let max_set: Vec<usize> = (0..d).chain(0..d).collect();

        // generate all the unique subsets
        let combis: Vec<Vec<usize>> = (1..(2 * d + 1))
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
            .collect();

        MultivarMomentAcc {
            ns: ns,
            pois: pois.to_owned(),
            nc: nc,
            d: d,
            n_traces: Array1::<u64>::zeros((nc,)),
            moments: Array3::<f64>::zeros((nc, combis.len(), ns)),
            mean: Array3::<f64>::zeros((nc, d, ns)),
            combis: combis,
        }
    }

    pub fn merge_from_state(
        &mut self,
        moments_other: ArrayView3<f64>,
        means: ArrayView3<f64>,
        n_traces: ArrayView1<u64>,
    ) {
        // each row will contain one power of deltas
        let combis = &self.combis;
        let ns = self.ns;
        izip!(
            self.moments.outer_iter_mut(),
            self.mean.outer_iter_mut(),
            self.n_traces.iter(),
            moments_other.outer_iter(),
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
    /// Updates the current estimation with fresh traces.
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let d = self.d;
        let mut moments_other = Array3::<f64>::zeros((self.nc, self.combis.len(), self.ns));
        let mut prod = Array2::<f64>::zeros((self.combis.len(), self.ns));
        let precomp_loc: Vec<usize> = self
            .combis
            .iter()
            .filter(|x| x.len() > 1)
            .map(|x| {
                if x.len() == 2 {
                    0
                } else {
                    let mut tmp = x.clone();
                    tmp.remove(0);
                    self.combis.iter().position(|y| tmp == *y).unwrap() - self.d
                }
            })
            .collect();
        //
        // STEP 1: 2-passes algorithm to compute center sum of powers
        //

        // STEP 1.1: process the all traces.
        // Compute the mean per class on the all traces
        let mut sum = Array2::<i64>::zeros((self.nc, traces.shape()[1]));
        let mut n_traces = Array1::<u64>::zeros(self.nc);

        for (trace, class) in traces.outer_iter().zip(y.iter()) {
            n_traces[*class as usize] += 1;
            let mut s = sum.slice_mut(s![*class as usize, ..]);
            s.zip_mut_with(&trace, |s, t| {
                *s += *t as i64;
            });
        }
        let n = n_traces.mapv(|x| x as f64);
        let mut mean = sum.mapv(|x| x as f64);
        mean.axis_iter_mut(Axis(1)).for_each(|mut m| m /= &n);

        // for each trace:
        //  1. center it (t - mu)
        //  2. re-order the traces for each of the pois
        let mut ct = Array1::<f64>::zeros((traces.shape()[1],));
        izip!(traces.axis_iter(Axis(0)), y.iter()).for_each(|(t, y)| {
            // prod according to poi for first (t-mu)
            center_trace(ct.view_mut(), t.view(), mean.slice(s![*y as usize, ..]));
            let mut to_update = moments_other.slice_mut(s![*y as usize, .., ..]);

            // re-order according to pois.
            for i in 0..d {
                let mut c = prod.slice_mut(s![i, ..]);
                let mut tmp = to_update.slice_mut(s![i,..]);
                c.zip_mut_with(&self.pois.slice(s![i, ..]), |c, p| *c = ct[*p as usize]);
                tmp += &c;
            }

            // compute the product for all the higher order combinations
            let (ct, mut higher_order) = prod.view_mut().split_at(Axis(0), self.d);
            let (_, mut to_update) = to_update.view_mut().split_at(Axis(0), self.d);
            let (_, higher_combi) = self.combis.split_at(self.d);
            izip!(
                (0..higher_order.len()),
                higher_combi.iter(),
                precomp_loc.iter(),
                to_update.axis_iter_mut(Axis(0)),
            )
            .for_each(|(i, combi, precomp_loc,to_update)| {
                let (tmp,mut hf) = higher_order.view_mut().split_at(Axis(0),i);
                let h = hf.slice_mut(s![0,..]);
                if combi.len() == 2 {
                    add_prod(to_update,h,ct.slice(s![combi[0], ..]),ct.slice(s![combi[1], ..]));
                } else {
                    add_prod(to_update,h,ct.slice(s![combi[0], ..]),tmp.slice(s![*precomp_loc, ..]));
                }
            });

            // add this combination
        });

        // STEP 2: merge this batch
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
        self.merge_from_state(moments_other.view(), mapped_means.view(), n_traces.view());
    }
}

pub struct MTtest {
    /// order of the test
    d: usize,
    /// Number of samples per trace
    ns: usize,
    /// Vector of Moment accumulators
    accumulators: Vec<MultivarMomentAcc>,
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

        let ns = pois.shape()[1];
        // number of required accumulators
        let n_batches = ((ns as f64) / (NS_BATCH as f64)).ceil() as usize;
        let accumulators: Vec<MultivarMomentAcc> = (0..n_batches)
            .map(|x| {
                let l = std::cmp::min(ns - (x * NS_BATCH), NS_BATCH);
                MultivarMomentAcc::new(pois.slice(s![.., (NS_BATCH * x)..(NS_BATCH * x + l)]), 2)
            })
            .collect();

        MTtest {
            d: d,
            ns: ns,
            accumulators: accumulators,
        }
    }
    /// Update the Ttest state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (n,)
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        for acc in &mut self.accumulators {
            acc.update(traces, y);
        }
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
            let mus = acc.moments.slice(s![.., id, ..]);
            mu1.assign(&(&mus.slice(s![0, ..]) / n1));
            mu2.assign(&(&mus.slice(s![1, ..]) / n2));

            let mut combi: Vec<usize> = (0..self.d).chain(0..self.d).collect();
            combi.sort();
            let id = acc.combis.iter().position(|x| *x == combi).unwrap();
            let vars = acc.moments.slice(s![.., id, ..]);
            var1.assign(&(&vars.slice(s![0, ..]) / n1));
            var2.assign(&(&vars.slice(s![1, ..]) / n2));

            if self.d > 2 {
                for j in 0..self.d {
                    let combi: Vec<usize> = vec![j, j];
                    let id = acc.combis.iter().position(|x| *x == combi).unwrap();
                    let mus = acc.moments.slice(s![.., id, ..]);
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

#[inline(always)]
pub fn center_trace(to: ArrayViewMut1<f64>, ti: ArrayView1<i16>, m: ArrayView1<f64>) {
    let to = to.into_slice().unwrap();
    let ti = ti.to_slice().unwrap();
    let m = m.to_slice().unwrap();
    izip!(to.iter_mut(), ti.iter(), m.iter()).for_each(|(to, ti, m)| {
        *to = *ti as f64 - *m;
    });
}

#[inline(always)]
pub fn add_prod(to1: ArrayViewMut1<f64>,to2: ArrayViewMut1<f64>, v1: ArrayView1<f64>, v2: ArrayView1<f64>) {
    let to1 = to1.into_slice().unwrap();
    let to2 = to2.into_slice().unwrap();
    let v1 = v1.to_slice().unwrap();
    let v2 = v2.to_slice().unwrap();
    izip!(to1.iter_mut(), to2.iter_mut(), v1.iter(), v2.iter()).for_each(|(to1, to2, v1, v2)| {
        *to2 = *v1 * *v2;
        *to1 += *to2;     
    });
}
