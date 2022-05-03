use itertools::{izip, Itertools};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

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
            moments_other.outer_iter(),
            means.outer_iter(),
            self.n_traces.iter(),
            n_traces.iter()
        )
        .for_each(|(mut cs1, mut u1, cs2, u2, n1, n2)| {
            let n1 = *n1 as f64;
            let n2 = *n2 as f64;
            let n = n1 + n2;
            if n1 == 0.0 {
                cs1.assign(&cs2);
                u1.assign(&u2);
            } else {
                let delta = &u2 - &u1;

                let mut prod1 = Array1::<f64>::ones((ns,));
                let mut prod2 = Array1::<f64>::ones((ns,));

                // update all the combinations one by one.
                for (i, combi) in combis.iter().enumerate() {
                    // split cs between the inputs and the outputs
                    let (cs1_smaller, mut cs1_larger) = cs1.view_mut().split_at(Axis(0), i);
                    let (cs2_smaller, cs2_larger) = cs2.view().split_at(Axis(0), i);
                    let mut cs1_larger = cs1_larger.slice_mut(s![0 as usize, ..]);

                    cs1_larger += &cs2_larger.slice(s![0, ..]);

                    for k in 2..combi.len() {
                        for set in combi.into_iter().combinations(k) {
                            let mut set: Vec<usize> = set.into_iter().map(|x| *x).collect();
                            let id = combis.iter().position(|x| *x == set).unwrap();

                            prod1.fill(1.0);
                            prod2.fill(1.0);
                            // product of missing values in set.
                            for x in combi.iter() {
                                if set.contains(x) {
                                    set.remove(set.iter().position(|y| *x == *y).unwrap());
                                } else {
                                    prod1 *= &(&delta.slice(s![*x, ..]) * (-n2 / n));
                                    prod2 *= &(&delta.slice(s![*x, ..]) * (-n1 / n));
                                }
                            }
                            prod1 *= &cs1_smaller.slice(s![id, ..]);
                            prod2 *= &cs2_smaller.slice(s![id, ..]);
                            cs1_larger += &prod1;
                            cs1_larger += &prod2;
                        }
                    }
                    prod1.fill((-n2 / n).powi(combi.len() as i32) * n1);
                    prod2.fill((-n1 / n).powi(combi.len() as i32) * n2);
                    for x in combi.into_iter() {
                        prod1 *= &delta.slice(s![*x, ..]);
                        prod2 *= &delta.slice(s![*x, ..]);
                    }
                    cs1_larger += &prod1;
                    cs1_larger += &prod2;
                }
                u1 += &((&u2 - &u1) * (n1 / n));
            }
        });
        self.n_traces += &n_traces;
    }
    /// Updates the current estimation with fresh traces.
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        let d = self.d;
        let mut moments_other = Array3::<f64>::zeros((self.nc, self.combis.len(), self.ns));
        let mut prod = Array2::<f64>::zeros((self.combis.len(), self.ns));
        //
        // STEP 1: 2-passes algorithm to compute center sum of powers
        //

        // STEP 1.1: process the all traces.
        // Compute the mean per class on the all traces
        let mut sum = Array2::<u64>::zeros((self.nc, traces.shape()[1]));
        let mut n_traces = Array1::<u64>::zeros(self.nc);
        for (trace, class) in traces.outer_iter().zip(y.iter()) {
            n_traces[*class as usize] += 1;
            let mut s = sum.slice_mut(s![*class as usize, ..]);
            s.zip_mut_with(&trace, |s, t| {
                *s += *t as u64;
            });
        }
        let n = n_traces.mapv(|x| x as f64);
        let mut mean = sum.mapv(|x| x as f64);
        mean.axis_iter_mut(Axis(1)).for_each(|mut m| m /= &n);

        // for each trace:
        //  1. center it (t - mu)
        //  2. re-order the traces for each of the pois
        izip!(traces.axis_iter(Axis(0)), y.iter()).for_each(|(t, y)| {
            // prod according to poi for first (t-mu)
            let ct = t.mapv(|x| x as f64) - &mean.slice(s![*y as usize, ..]);

            // re-order according to pois.
            for i in 0..d {
                let mut c = prod.slice_mut(s![i, ..]);
                c.zip_mut_with(&self.pois.slice(s![i, ..]), |c, p| *c = ct[*p as usize]);
            }

            // compute the product for all the higher order combinations
            let (ct, mut higher_order) = prod.view_mut().split_at(Axis(0), self.d);
            let (_, higher_combi) = self.combis.split_at(self.d);
            izip!(higher_order.axis_iter_mut(Axis(0)), higher_combi.iter()).for_each(
                |(mut h, combi)| {
                    h.fill(1.0);
                    for c in combi.iter() {
                        h *= &ct.slice(s![*c, ..]);
                    }
                },
            );

            // add this combination
            let mut to_update = moments_other.slice_mut(s![*y as usize, .., ..]);
            to_update += &prod;
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
