// state i64, acc local i32, dyn bit width
//
//! Estimation for signal-to-noise ratio.
//!
//! An estimation of SNR is represented with a SNR struct. Calling update allows
//! to update the SNR state with fresh measurements. get_snr returns the current value
//! of the estimate.
//! The SNR can be computed for np independent random variables and the same measurements.
//! The measurements are expected to be of length ns. The random variable values must be
//! included in [0,nc[.
//!
//! The SNR estimation algorithm works as follows. We consider a single point in the trace (all
//! points are treated in the same way).
//! For every $i=0,\dots,nc-1$ ($nc$ is the number of classes), let $x\_{i,j}$ (for
//! $j=0,\dots,n\_i-1$) be all the leakages of class $i$.
//! Let $\mu\_i = \sum\_{j=0}^{n\_i-1} x_{i,j}/n\_i$ and $n = \sum\{i=0}^{nc-1} n\_i$.
//! Moreover, let
//! $S\_i = \sum\_j x\_{i,j}$, $S = \sum\_i S\_i$, $SS\_i = \sum\_j x\_{i,j}^2$ and $SS = \sum\_i
//! SS\_i$.
//!
//! We compute $SNR = Sig/No$, where
//!
//! $$
//! No
//! = \sum\_{i=0}^{nc-1} \sum\_{j=0}^{n\_i-1} 1/(n-nc) (x\_{i,j}-\mu\_i)^2
//!  = \sum\_{i,j} 1/(n-nc) * x_{i,j}^2 - \sum\_i 1/(n-nc)/n\_i (\sum\_j x\_{i,j})^2
//!  = 1/(n-nc) (SS - \sum\_i 1/n\_i S\_i^2)
//!  = 1/(n(n-nc)) (n SS - \sum\_i n/n\_i S\_i^2)
//! $$
//!
//! so that
//!
//! - the $No$ estimator is non-biased, and
//! - we use a [pooled variance](https://en.wikipedia.org/wiki/Pooled_variance) to ensure an
//! accurate estimation when the samples in the classes are not balanced.
//!
//! For the signal, we proceed similarly:
//!
//! $$
//! Sig = \sum\_i n\_i/(n-nc) (mu\_i-mu)^2
//!     = \sum\_i n\_i/(n-nc) (S\_i/n\_i-S/n)^2
//!     = \sum\_i n\_i/(n-nc) \left S\_i^2/n\_i^2 -2S\_i/n\_i S/n + S^2/n^2 \right)
//!     = 1/(n(n-nc)) \left(\sum\_i n/n\_i S\_i^2 - S^2\right)
//! $$
//!
//! For both $No$ and $Sig$, when some $n\_i$ is zero, it is left out from the sums (avoiding the
//! need to compute $1/n\_i$) and $nc$ is consequently decreased.
//!
//!
//! Regarding the implementation, we have to compute $SS$, $S\_i$, $n\_i$ and $nc$, from which $S$
//! and $n$ can be easily derived.
//! Assuming 16-bit data, and $n<2^32$, we can store $SS$ and $S\_i$ on 64-bit integers (and $S\_i$
//! could even be on 32-bit if $n<2^16$, or for temporary accumulators).
//! For the final computation, for $No$, we can have $S\_i^2$ on 128-bit integer (small loss of
//! performance, should not be too costly), then $S\_i^2/n\_i$ on 64-bit integer.
//! For $Sig$, $(n S\_i^2) / n\_i$ can be computed on 128-bit, as well as $S^2$.

use crate::ScalibError;
use itertools::izip;
use ndarray::{s, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::lvar::{self, AccType, LVar};

/// SNR state.
/// This allows to estimate the mean and the variance for each of the classes which are
/// needed for SNR.
#[derive(Debug)]
pub struct SNR<T: AccType> {
    acc: LVar<T>,
}

impl<T: AccType> SNR<T> {
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// nv: number of independent random variable for which SNR must be estimated
    pub fn new(nc: usize, ns: usize, nv: usize) -> Self {
        Self {
            acc: LVar::new(nc, ns, nv),
        }
    }

    /// Update the SNR state with n fresh traces
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

    /// Generate the actual SNR metric based on the current state.
    /// return array axes (variable, samples in trace)
    pub fn get_snr(&self) -> Array2<f64> {
        let mut snr = Array2::<f64>::zeros((self.acc.nv(), self.acc.ns()));
        // on chunks of samples
        (
            self.acc.sum().axis_iter(Axis(0)),
            self.acc.sum_square().axis_iter(Axis(0)),
            snr.axis_chunks_iter_mut(Axis(1), lvar::SIMD_SIZE),
        )
            .into_par_iter()
            .for_each(|(sum, sum_square, mut snr)| {
                let sum_square: &[i64; lvar::SIMD_SIZE] = sum_square.into_scalar();
                let general_sum =
                    sum.slice(s![0usize, ..])
                        .iter()
                        .fold([0i64; lvar::SIMD_SIZE], |mut acc, s| {
                            for (acc, s) in izip!(acc.iter_mut(), s.iter()) {
                                // no overflow: sample on 16 bits, at most 2^32 traces
                                *acc += T::acc2i64(*s);
                            }
                            acc
                        });
                let mut general_sum_sq = [0i128; lvar::SIMD_SIZE];
                for (sq, s) in izip!(general_sum_sq.iter_mut(), general_sum.iter()) {
                    let s = *s as i128;
                    *sq = s * s;
                }
                // on variables
                izip!(
                    sum.axis_iter(Axis(0)),
                    self.acc.n_samples().axis_iter(Axis(0)),
                    snr.axis_iter_mut(Axis(0))
                )
                .for_each(|(sum, n_samples, snr)| {
                    compute_snr::<T>(
                        sum.to_slice().unwrap(),
                        n_samples.to_slice().unwrap(),
                        sum_square,
                        &general_sum_sq,
                        self.acc.tot_n_samples(),
                        snr.into_slice().unwrap(),
                    );
                });
            });
        snr
    }
}

#[inline(never)]
fn compute_snr<T: AccType>(
    sum: &[[T::SumAcc; lvar::SIMD_SIZE]],
    n_samples: &[u32],
    sum_square: &[i64; lvar::SIMD_SIZE],
    general_sum_sq: &[i128; lvar::SIMD_SIZE],
    n: u32,
    snr: &mut [f64],
) {
    let sum_square_class =
        izip!(sum.iter(), n_samples.iter()).fold([0i128; lvar::SIMD_SIZE], |mut acc, (s, ns)| {
            for (acc, s) in izip!(acc.iter_mut(), s.iter()) {
                if *ns != 0 {
                    let s = T::acc2i64(*s) as i128;
                    // No overflow: s is on <= (16+32) bit (signed), n is on 32-bit therefore, s*s
                    // in on < 96 bits (signed), and n*s*s is on <128 bits (signed)
                    // TODO optimize this bottleneck, the division is 75% exec. time (e.g.
                    // use libdivide)
                    *acc += s * s * (n as i128) / (*ns as i128);
                }
            }
            acc
        });
    let l = snr.len();
    izip!(
        sum_square_class[..l].iter(),
        general_sum_sq[..l].iter(),
        sum_square[..l].iter(),
        snr.iter_mut()
    )
    .for_each(|(sum_square_class, general_sum_sq, sum_square, snr)| {
        let sum_square = *sum_square as i128;
        let signal = sum_square_class - general_sum_sq;
        let noise = (n as i128) * sum_square - sum_square_class;
        *snr = (signal as f64) / (noise as f64);
    });
}
