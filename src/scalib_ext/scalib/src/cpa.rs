use crate::ScalibError;
use itertools::izip;
use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, Axis};
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
        let mut res = Array3::<f64>::zeros((self.acc.nv(), self.acc.nc(), self.acc.ns()));
        res
    }
}

const SIMD_SIZE: usize = 8;
type SIMD_ACC = [f64; SIMD_SIZE];

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

/// Compute correlation for a given var and a sample block
fn correlation_internal<const KEY_BLOCK: usize, T: AccType>(
    sums: &[[T::SumAcc; SIMD_SIZE]],
    models: &[SIMD_ACC],
    nc: usize,
    mut res: ArrayViewMut1<SIMD_ACC>,
) {
    // TODO: variances
    // Covariance:
    // TODO: do not re-allocate sums at each call (move allocation to caller function).
    let sums = sums
        .iter()
        .map(|x| x.map(|y| T::acc2i64(y) as f64))
        .collect::<Vec<_>>();
    if KEY_BLOCK < nc {
        let start_blocks = nc % KEY_BLOCK;
        if start_blocks != 0 {
            ip_core::<KEY_BLOCK, T>(&sums, models, nc, 0, res.slice_mut(s![0..KEY_BLOCK]));
        }
        for (i, res) in res
            .slice_mut(s![start_blocks..])
            .exact_chunks_mut((KEY_BLOCK,))
            .into_iter()
            .enumerate()
        {
            let start_key = start_blocks + KEY_BLOCK * i;
            ip_core::<KEY_BLOCK, T>(&sums, models, nc, start_key, res);
        }
    } else {
        for (start_key, res) in res.exact_chunks_mut((1,)).into_iter().enumerate() {
            ip_core::<1, T>(&sums, models, nc, start_key, res);
        }
    }
}

/// Compute inner product between sums and models
/// for 1 SIMD word of samples and one block of keys.
fn ip_core<const NKEYS: usize, T: AccType>(
    // sums associated to each label
    sums: &[SIMD_ACC],
    models: &[SIMD_ACC],
    nc: usize,
    key_start: usize,
    mut res: ArrayViewMut1<SIMD_ACC>,
) {
    assert_eq!(res.shape()[0], NKEYS);
    res.fill([0.0; SIMD_SIZE]);
    for label in 0..nc {
        let sums = sums[label];
        for (res, key) in res.iter_mut().zip(key_start..(key_start + NKEYS)) {
            let intermediate = key ^ label;
            // TODO SAFETY
            let model = unsafe { models.get_unchecked(intermediate) };
            for i in 0..SIMD_SIZE {
                res[i] = sums[i].mul_add(model[i], res[i]);
            }
        }
    }
}
