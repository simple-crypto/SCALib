use itertools::{izip, Itertools};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

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

    /// Updates the current estimation with fresh traces.
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView1<u16>) {
        println!("Welcome to update");
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
        self.moments.assign(&moments_other);
        let pois = &self.pois;
        izip!(self.mean.outer_iter_mut(),
                mean.outer_iter()
        ).for_each(|(mut to_update,mean)|{
                izip!(to_update.outer_iter_mut(),pois.outer_iter())
                    .for_each(|(mut to_update, pois)|{
            to_update.zip_mut_with(&pois,|x,p| *x = mean[*p as usize]);
                    });
        });
    }
}
