use crate::ScalibError;
use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, Axis};
use rayon::prelude::*;

use crate::lvar::{AccType, LVar};
#[derive(Debug)]
pub struct CPA<T: AccType> {
    acc: LVar<T>,
}

impl<T: AccType> CPA<T> {
    /// Create a new SNR state.
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
    /// If this errors, the SNR object should not be used anymore.
    /// traces and y must be in standard C order
    pub fn update(
        &mut self,
        traces: ArrayView2<i16>,
        y: ArrayView2<u16>,
        config: &crate::Config,
    ) -> Result<(), ScalibError> {
        self.acc.update(traces, y, config)
    }

    /// Generate the actual CPA based on the current state.
    /// return array axes (variable, key class, samples in trace)
    /// models: (variable, intermediate variable class, samples in trace)
    /// (assumes intermediate variable = key ^ label)
    pub fn compute_cpa(&self, models: ArrayView3<f64>) -> Array3<f64> {
        const KEY_BLOCK: usize = 8;
        let mut res = Array3::<f64>::zeros((self.acc.nv(), self.acc.nc(), self.acc.ns()));
        let tot_sums = self.acc.tot_sum();
        // Iterate over variables
        (
            res.axis_iter_mut(Axis(0)),
            models.axis_iter(Axis(0)),
            self.acc.sum().axis_iter(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(mut res, models, sums)| {
                // Iterate blocks of SIMD_SIZE samples in trace.
                (
                    res.axis_chunks_iter_mut(Axis(1), SIMD_SIZE),
                    models.axis_chunks_iter(Axis(1), SIMD_SIZE),
                    sums.axis_iter(Axis(1)),
                    tot_sums.axis_iter(Axis(0)),
                    self.acc.sum_square().axis_iter(Axis(0)),
                )
                    .into_par_iter()
                    .for_each_init(
                        || CorrelationTmp::new(self.acc.nc()),
                        |tmp, (res, models, sums, tot_sums, sums_squares)| {
                            correlation_internal::<KEY_BLOCK, T>(
                                *tot_sums.into_scalar(),
                                *sums_squares.into_scalar(),
                                sums,
                                models,
                                self.acc.tot_n_samples(),
                                self.acc.nc(),
                                res,
                                tmp,
                            );
                        },
                    )
            });
        res
    }
}

const SIMD_SIZE: usize = 8;
type SimdAcc = [f64; SIMD_SIZE];

// Trivial algo:
// for (var, sample block):
//      for sample:
//        compute model variance
//        compute trace variance
//      for key block:
//          initialize accumulators
//          for label:
//              load sums for label
//              for key (unrolled):
//                  intermediate = key ^ label
//                  for sample (SIMD):
//                      acc += sum * model[intermediate]

fn sumarray<const N: usize, T: std::ops::Add<T, Output = T> + Copy>(
    a: [T; N],
    b: &[T; N],
) -> [T; N] {
    std::array::from_fn(|i| a[i] + b[i])
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
            cmodels: vec![[0.0; SIMD_SIZE]; nc as usize],
            csums: vec![[0.0; SIMD_SIZE]; nc as usize],
            res_tmp: vec![[0.0; SIMD_SIZE]; nc as usize],
        }
    }
}

/// Compute correlation for a given var and a sample block
fn correlation_internal<const KEY_BLOCK: usize, T: AccType>(
    glob_sums: [i64; SIMD_SIZE],
    sums_squares: [i64; SIMD_SIZE],
    sums: ArrayView1<[T::SumAcc; SIMD_SIZE]>,
    models: ArrayView2<f64>,
    n: u32,
    nc: usize,
    mut res: ArrayViewMut2<f64>,
    tmp: &mut CorrelationTmp,
) {
    assert!(models.shape()[1] <= SIMD_SIZE);
    assert_eq!(models.strides()[1], 1);
    let CorrelationTmp {
        cmodels,
        csums,
        res_tmp,
    } = tmp;
    let inv_n = 1.0 / (n as f64);
    let glob_means = glob_sums.map(|x| (x as f64) * inv_n);
    for (csums, sums) in csums.iter_mut().zip(sums.iter()) {
        *csums = std::array::from_fn(|i| (T::acc2i64(sums[i]) as f64) - glob_means[i]);
    }
    let mut sum_models = [0.0; SIMD_SIZE];
    let mut sum_models_v =
        ArrayViewMut1::from(&mut sum_models.as_mut_slice()[0..models.shape()[1]]);
    for m in models.axis_iter(Axis(0)) {
        sum_models_v += &m;
    }
    let means_models = sum_models.map(|x| x * inv_n);
    for (cmodels, models) in cmodels.iter_mut().zip(models.axis_iter(Axis(0))) {
        let models = models.as_slice().unwrap();
        for (cm, m) in cmodels.iter_mut().zip(models.iter()) {
            *cm = *m;
        }
        *cmodels = std::array::from_fn(|i| cmodels[i] - means_models[i]);
    }
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

/// Variance of the model (cmodels: centered models, nc: number of classes)
fn model_variance(cmodels: &[SimdAcc], nc: f64) -> SimdAcc {
    cmodels
        .iter()
        .fold([0.0; SIMD_SIZE], |a, x| sumarray(a, &x.map(|y| y * y)))
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
    assert_eq!(res.len(), NKEYS);
    res.fill([0.0; SIMD_SIZE]);
    for label in 0..nc {
        let sums = sums[label];
        for (res, key) in res.iter_mut().zip(key_start..(key_start + NKEYS)) {
            let intermediate = key ^ label;
            // TODO SAFETY check before this function (for XOR, need nc to be power of 2 and all
            // correct sizes).
            let model = unsafe { models.get_unchecked(intermediate) };
            for i in 0..SIMD_SIZE {
                res[i] = sums[i].mul_add(model[i], res[i]);
            }
        }
    }
}
