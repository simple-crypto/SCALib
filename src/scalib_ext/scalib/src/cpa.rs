use crate::{lvar::SIMD_SIZE, Result, ScalibError};
use itertools::izip;
use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, Axis};
use rayon::prelude::*;

use crate::lvar::{self, AccType, LVar};

type SimdAcc = [f64; lvar::SIMD_SIZE];
const SIMD_ZERO: SimdAcc = [0.0; SIMD_SIZE];

#[derive(Debug)]
pub struct CPA<T: AccType> {
    acc: LVar<T>,
}

impl<T: AccType> CPA<T> {
    /// Create a new CPA state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// nv: number of independent random variable for which CPA must be estimated
    pub fn new(nc: usize, ns: usize, nv: usize) -> Self {
        Self {
            acc: LVar::new(nc, ns, nv),
        }
    }

    /// Update the CPA state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    /// If this errors, the CPA object should not be used anymore.
    /// traces and y must be in standard C order
    pub fn update(
        &mut self,
        traces: ArrayView2<i16>,
        y: ArrayView2<u16>,
        config: &crate::Config,
    ) -> Result<()> {
        self.acc.update(traces, y, config)
    }

    /// Generate the actual CPA based on the current state.
    /// return array axes (variable, key class, samples in trace)
    /// models: (variable, intermediate variable class, samples in trace)
    /// (assumes intermediate variable = key ^ label)
    pub fn compute_cpa(&self, models: ArrayView3<f64>) -> Result<Array3<f64>> {
        const KEY_BLOCK: usize = 8;
        let expected_mdim = (self.acc.nv(), self.acc.nc(), self.acc.ns());
        if models.dim() != expected_mdim {
            return Err(ScalibError::CpaMShape {
                dim: models.dim(),
                expected: expected_mdim,
            });
        }
        let mut res = Array3::<f64>::zeros((self.acc.nv(), self.acc.nc(), self.acc.ns()));
        // Sum of all traces, Array1<[i64; SIMD_SIZE]>
        let tot_sums = self.acc.tot_sum();
        // Iterate over variables
        (
            res.axis_iter_mut(Axis(0)),
            models.axis_iter(Axis(0)),
            self.acc.sum().axis_iter(Axis(1)),
            self.acc.n_samples().axis_iter(Axis(0)),
        )
            .into_par_iter()
            .for_each_init(
                || vec![0.0; self.acc.nc()],
                |n_samples_f, (mut res, models, sums, n_samples)| {
                    // Shapes:
                    // res (nc, ns)
                    // models (nc, ns)
                    // sums (ceil(ns/SIMD_SIZE), nc)
                    // nsamples (nc)
                    assert!(res.dim() == models.dim());
                    assert!(res.shape()[0] == n_samples.len());
                    assert!(sums.shape()[0] == res.shape()[1].div_ceil(SIMD_SIZE));
                    for (nsf, ns) in n_samples_f.iter_mut().zip(n_samples.iter()) {
                        *nsf = *ns as f64;
                    }
                    // Iterate blocks of SIMD_SIZE samples in trace.
                    (
                        // (nc, SIMD_SIZE)
                        res.axis_chunks_iter_mut(Axis(1), lvar::SIMD_SIZE),
                        // (nc, SIMD_SIZE)
                        models.axis_chunks_iter(Axis(1), lvar::SIMD_SIZE),
                        // len = nc
                        sums.axis_iter(Axis(0)),
                        tot_sums.axis_iter(Axis(0)),
                        self.acc.sum_square().axis_iter(Axis(0)),
                    )
                        .into_par_iter()
                        .for_each_init(
                            || CorrelationTmp::new(self.acc.nc()),
                            |tmp, (res, models, sums, tot_sums, sums_squares)| {
                                // Shapes:
                                // res: (nc, SIMD_SIZE)
                                // models: (nc, SIMD_SIZE)
                                // sums: (nc,)
                                assert!(models.dim() == res.dim());
                                assert!(sums.len() == res.shape()[0]);
                                assert!(tot_sums.len() == 1);
                                correlation_internal::<KEY_BLOCK, T>(
                                    *tot_sums.into_scalar(),
                                    *sums_squares.into_scalar(),
                                    sums,
                                    &n_samples_f,
                                    models,
                                    self.acc.tot_n_samples(),
                                    self.acc.nc(),
                                    res,
                                    tmp,
                                );
                            },
                        )
                },
            );
        Ok(res)
    }
}

#[derive(Debug)]
struct CorrelationTmp {
    cmodels: Vec<SimdAcc>,
    csums: Vec<SimdAcc>,
    res_tmp: Vec<SimdAcc>,
}
impl CorrelationTmp {
    fn new(nc: usize) -> Self {
        Self {
            cmodels: vec![SIMD_ZERO; nc as usize],
            csums: vec![SIMD_ZERO; nc as usize],
            res_tmp: vec![SIMD_ZERO; nc as usize],
        }
    }
}

/// Compute correlation for a given var and a sample block
fn correlation_internal<const KEY_BLOCK: usize, T: AccType>(
    glob_sums: [i64; lvar::SIMD_SIZE],
    sums_squares: [i64; lvar::SIMD_SIZE],
    sums: ArrayView1<[T::SumAcc; lvar::SIMD_SIZE]>,
    n_samples: &[f64],
    models: ArrayView2<f64>,
    n: u32,
    nc: usize,
    mut res: ArrayViewMut2<f64>,
    tmp: &mut CorrelationTmp,
) {
    // x == data, xi == data with class i
    // n == number of traces, ni == number of traces with class i
    // Sx(.) == sum over all traces
    // Sxi(.) == sum over traces of class i
    // mu = Sx(x)/n (average)
    // mui = Sxi(xi)/ni
    //
    // csums = Sum( xi - mu)
    //       = Sum( xi - Sx / n)
    //       = Sxi - ni * Sx / n
    assert!(models.shape()[1] <= lvar::SIMD_SIZE);
    assert_eq!(models.strides()[1], 1);
    let CorrelationTmp {
        cmodels,
        csums,
        res_tmp,
    } = tmp;
    // Compute the centered traces, per class
    let inv_n = 1.0 / (n as f64);
    let glob_means = glob_sums.map(|x| (x as f64) * inv_n);
    for (csums, sums, n_samples) in izip!(csums.iter_mut(), sums.iter(), n_samples.iter()) {
        *csums = std::array::from_fn(|i| (T::acc2i64(sums[i]) as f64) - n_samples * glob_means[i]);
    }
    // Compute the centered models, per class
    compute_cmodels::<T>(models, cmodels);
    // Covariance scaled by n:
    // if nc is too low, handle one-by-one.
    if nc < KEY_BLOCK {
        for (start_key, tmp) in res_tmp.chunks_exact_mut(1).enumerate() {
            ip_core::<1, T>(&csums, &cmodels, nc, start_key, tmp);
        }
    } else {
        // Split nc in blocks of size KEY_BLOCK.
        // If nc is not a multiple of KEY_BLOCK, we handle the first block separately, then handle
        // all remaining blocks staring a index nc % KEY_BLOCK.
        // The first block starts at index 0 and overlaps with the second block, which is not an
        // issue since the computation of covariance is idempotent.
        let start_blocks = nc % KEY_BLOCK;
        if start_blocks != 0 {
            ip_core::<KEY_BLOCK, T>(&csums, &cmodels, nc, 0, &mut res_tmp[0..KEY_BLOCK]);
        }
        for (i, tmp) in res_tmp[start_blocks..]
            .chunks_exact_mut(KEY_BLOCK)
            .enumerate()
        {
            let start_key = start_blocks + KEY_BLOCK * i;
            ip_core::<KEY_BLOCK, T>(&csums, &cmodels, nc, start_key, tmp);
        }
    }
    // Variances
    let var_model = model_variance(&cmodels, nc as f64);
    let var_data = data_variance(&glob_sums, &sums_squares, n);
    // n appears in denominator because tmp is a non-scaled inner product.
    let inv_denom: SimdAcc =
        std::array::from_fn(|i| 1.0 / ((n as f64) * (var_model[i] * var_data[i]).sqrt()));
    // Correlation computation & writeback.
    for (mut res, tmp) in res.axis_iter_mut(Axis(0)).zip(res_tmp.iter()) {
        let tmp: SimdAcc = std::array::from_fn(|i| tmp[i] * inv_denom[i]);
        for (res, tmp) in res.iter_mut().zip(tmp.iter()) {
            *res = *tmp;
        }
    }
}

// For a single var, compute the centered models
// models: (nc, SIMD_SIZE)
// cmodels: vec![(SIMD_SIZE, ) ; nc]
fn compute_cmodels<T: AccType>(models: ArrayView2<f64>, cmodels: &mut [SimdAcc]) {
    assert_eq!(models.shape()[0], cmodels.len());
    assert!(models.shape()[1] <= lvar::SIMD_SIZE);
    assert_eq!(models.strides()[1], 1);
    let nc = models.shape()[0];
    let inv_nc = 1.0 / (nc as f64);
    let mut sum_models = SIMD_ZERO;
    let mut sum_models_v =
        ArrayViewMut1::from(&mut sum_models.as_mut_slice()[0..models.shape()[1]]);
    // Compute the sum of model, point-wise, corresponding to the <=SIMD_SIZE samples.
    for m in models.axis_iter(Axis(0)) {
        sum_models_v += &m;
    }
    // Mean model, per sample, as the sum of all the model divided by the amount of
    // classes.
    let means_models = sum_models.map(|x| x * inv_nc);
    for (cmodels, models) in cmodels.iter_mut().zip(models.axis_iter(Axis(0))) {
        // Shapes
        // cmodels: (SIMD_SIZE, ), one per class
        // models: (SIMD_SIZE, ), one per class
        let models = models.as_slice().unwrap();
        // Copy the models in cmodels
        for (cm, m) in cmodels.iter_mut().zip(models.iter()) {
            *cm = *m;
        }
        // Compute the centered model as the model minus the mean, for each sample
        *cmodels = std::array::from_fn(|i| cmodels[i] - means_models[i]);
    }
}

/// Variance of the model (cmodels: centered models, nc: number of classes)
fn model_variance(cmodels: &[SimdAcc], nc: f64) -> SimdAcc {
    cmodels
        .iter()
        .fold(SIMD_ZERO, |a, x| sumarray(a, &x.map(|y| y * y)))
        .map(|x| x / nc)
}

/// Variance of the traces (glob_sums: sum of the traces, sums_squares: sum of traces squared, n:
/// total number of traces).
fn data_variance(glob_sums: &[i64; SIMD_SIZE], sums_squares: &[i64; SIMD_SIZE], n: u32) -> SimdAcc {
    let nf = n as f64;
    let inv_n_sq = 1.0 / (nf * nf);
    // Var(x) = sum(x-mu)**2/n = sum(x**2)/n - mu**2 = (n*sum(x**2) - sum(x)**2)/n**2
    std::array::from_fn(|i| {
        let ss = sums_squares[i] as i128;
        let gs = glob_sums[i] as i128;
        let num_i128 = ((n as i128) * ss) - gs * gs;
        (num_i128 as f64) / inv_n_sq
    })
}

/// Compute inner product between sums and models
/// for 1 SIMD word of samples and one block of keys.
fn ip_core<const NKEYS: usize, T: AccType>(
    // sums associated to each label
    sums: &[SimdAcc],
    models: &[SimdAcc],
    nc: usize,
    key_start: usize,
    res: &mut [SimdAcc],
) {
    assert!(key_start + NKEYS <= nc); // Needed for safety, should be optimized out of caller loop by compiler.
    assert!(nc.is_power_of_two()); // Needed for safety, should be optimized out of caller loop by compiler.
    assert_eq!(models.len(), nc); // Needed for safety, should be optimized out of caller loop by compiler.
    assert_eq!(res.len(), NKEYS);
    res.fill(SIMD_ZERO);
    for label in 0..nc {
        let sums = sums[label];
        for (res, key) in res.iter_mut().zip(key_start..(key_start + NKEYS)) {
            let intermediate = key ^ label;
            // SAFETY: the asserts at the top of this function ensure that
            // key < nc and nc is a power of two, therefore, since label < nc, we know that
            // intermediate < nc, and nc is the length of model, so it's in bounds.
            let model = unsafe { models.get_unchecked(intermediate) };
            for i in 0..lvar::SIMD_SIZE {
                res[i] = sums[i].mul_add(model[i], res[i]);
            }
        }
    }
}

fn sumarray<const N: usize, T: std::ops::Add<T, Output = T> + Copy>(
    a: [T; N],
    b: &[T; N],
) -> [T; N] {
    std::array::from_fn(|i| a[i] + b[i])
}

#[cfg(test)]
mod tests_cpa {
    use super::*;
    use ndarray::{Array1, Array2};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256StarStar;

    #[test]
    fn test_model_variance() {
        let seed = 0;
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let test_cases = [(2, 1), (256, 1), (256, 8), (2, 8)];
        for (nc, ns) in test_cases {
            println!("nc: {nc}, ns: {ns}");
            assert!(ns <= SIMD_SIZE);
            let models = Array2::<f64>::random_using((nc, ns), Uniform::new(0.0, 1.0), &mut rng);
            let mut cmodels = vec![SIMD_ZERO; nc];
            compute_cmodels::<crate::lvar::AccType64bit>(models.view(), &mut cmodels);
            let variance = model_variance(&cmodels, nc as f64);
            let variance = variance.into_iter().take(ns).collect::<Array1<_>>();
            // FIXME change DDOF parameter ?
            let variance_test = models
                .axis_iter(Axis(1))
                .map(|x| x.var(0.0))
                .collect::<Array1<_>>();
            assert!(
            variance.relative_eq(&variance_test, 1e-8, 1e-5),
            "Model variance mismatch, \nvariance: {variance:#?}\nvariance_test:{variance_test:#?}\nmodels: {models:#?}\ncmodels: {cmodels:#?}"
        );
        }
    }
}
