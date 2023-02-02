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
use hytra::TrAdder;
use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayViewMut2, Axis, Zip};
use num_traits::{Bounded, PrimInt, Signed, WrappingAdd, Zero};
use rayon::prelude::*;
use std::convert::TryInto;

pub trait NativeInt: PrimInt + Signed + WrappingAdd + Send + Sync {}
impl<T: PrimInt + Signed + WrappingAdd + Send + Sync> NativeInt for T {}

pub trait SnrType {
    const UPDATE_SNR_CHUNK_SIZE: usize;
    type SumAcc;
    type Sample;
    fn sample2tmp(s: Self::Sample) -> i32;
    fn sample2i64(s: Self::Sample) -> i64;
    fn tmp2acc(s: i32) -> Self::SumAcc;
    fn acc2i64(acc: Self::SumAcc) -> i64;
}

const TRACES_CHUNK_SIZE: usize = 1024;

#[derive(Debug)]
pub struct SnrType64bit;
#[derive(Debug)]
pub struct SnrType32bit;

impl SnrType for SnrType64bit {
    const UPDATE_SNR_CHUNK_SIZE: usize = 1 << 13;
    type SumAcc = i64;
    type Sample = i16;
    #[inline(always)]
    fn tmp2acc(s: i32) -> Self::SumAcc {
        s as i64
    }
    #[inline(always)]
    fn acc2i64(acc: Self::SumAcc) -> i64 {
        acc
    }
    #[inline(always)]
    fn sample2tmp(s: Self::Sample) -> i32 {
        s as i32
    }
    #[inline(always)]
    fn sample2i64(s: Self::Sample) -> i64 {
        s as i64
    }
}
impl SnrType for SnrType32bit {
    const UPDATE_SNR_CHUNK_SIZE: usize = 1 << 13;
    type SumAcc = i32;
    type Sample = i16;
    #[inline(always)]
    fn tmp2acc(s: i32) -> Self::SumAcc {
        s as i32
    }
    #[inline(always)]
    fn acc2i64(acc: Self::SumAcc) -> i64 {
        acc as i64
    }
    #[inline(always)]
    fn sample2tmp(s: Self::Sample) -> i32 {
        s as i32
    }
    #[inline(always)]
    fn sample2i64(s: Self::Sample) -> i64 {
        s as i64
    }
}

/// SNR state. stores the sum and the sum of squares of the leakage for each of the class.
/// This allows to estimate the mean and the variance for each of the classes which are
/// needed for SNR.
#[derive(Debug)]
pub struct SNR<T = SnrType32bit>
where
    T: SnrType,
    T::SumAcc: NativeInt,
{
    /// Sum of all the traces corresponding to each of the classes. shape (ceil(ns/8),np,nc)
    sum: Array3<[T::SumAcc; 8]>,
    /// Sum of squares with shape (ceil(ns/8))
    /// (never overflows since samples are i16 and tot_n_samples <= u32::MAX)
    sum_square: Array1<[i64; 8]>,
    /// number of samples per class (np,nc)
    n_samples: Array2<u32>,
    /// number of independent variables
    np: usize,
    /// number of samples in a trace
    ns: usize,
    /// number of classes
    nc: u32,
    /// max sample bit width
    bit_width: u32,
    /// total number of accumulated traces
    tot_n_samples: u32,
}

impl<T> SNR<T>
where
    T: SnrType<Sample = i16> + std::fmt::Debug,
    T::SumAcc: NativeInt,
{
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which SNR must be estimated
    pub fn new(nc: usize, ns: usize, np: usize) -> Self {
        let ns8 = if ns % 8 == 0 { ns / 8 } else { ns / 8 + 1 };
        assert!(nc <= 1 << 16);
        SNR {
            sum: Array3::from_elem((ns8, np, nc), [Zero::zero(); 8]),
            sum_square: Array1::from_elem((ns8,), [0; 8]),
            n_samples: Array2::zeros((np, nc)),
            np,
            ns,
            nc: nc.try_into().expect("Too many classes"),
            bit_width: 1,
            tot_n_samples: 0,
        }
    }

    /// Update the SNR state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    /// If this errors, the SNR object should not be used anymore.
    /// traces and y must be in standard C order
    pub fn update(
        &mut self,
        traces: ArrayView2<T::Sample>,
        y: ArrayView2<u16>,
        config: &crate::Config,
    ) -> Result<(), ScalibError> {
        let n_it = (self.sum.shape()[0] as u64 + 3) / 4;
        crate::utils::with_progress(
            |it_cnt| self.update_internal(traces, y, it_cnt),
            n_it,
            "Update SNR",
            config,
        )
    }

    #[inline(never)]
    /// If this errors, the SNR object should not be used anymore.
    fn update_internal(
        &mut self,
        traces: ArrayView2<T::Sample>,
        y: ArrayView2<u16>,
        acc_ref: &TrAdder<u64>,
    ) -> Result<(), ScalibError> {
        assert_eq!(traces.shape()[0], y.shape()[1]);
        assert_eq!(traces.shape()[1], self.ns);
        assert_eq!(y.shape()[0], self.np);
        assert!(traces.is_standard_layout());
        assert!(y.is_standard_layout());
        let n_traces: u32 = traces.shape()[0]
            .try_into()
            .map_err(|_| ScalibError::SnrTooManyTraces)?;
        self.tot_n_samples = self
            .tot_n_samples
            .checked_add(n_traces)
            .ok_or(ScalibError::SnrTooManyTraces)?;
        let mut max_n_samples: u32 = 0;
        let nc = self.nc;
        let np = self.np;
        izip!(self.n_samples.outer_iter_mut(), y.outer_iter()).try_for_each(
            |(mut n_samples, y)| {
                y.into_iter().try_for_each(|y| {
                    if u32::from(*y) >= nc {
                        Err(ScalibError::SnrClassOutOfBound)
                    } else {
                        n_samples[*y as usize] += 1;
                        max_n_samples = std::cmp::max(max_n_samples, n_samples[*y as usize]);
                        Ok(())
                    }
                })
            },
        )?;
        let sample_bits_used_msk = (
            self.sum.axis_chunks_iter_mut(Axis(0), 32 / 8),
            self.sum_square.axis_chunks_iter_mut(Axis(0), 32 / 8),
            traces.axis_chunks_iter(Axis(1), 32),
        )
            .into_par_iter()
            .map_init(
                || {
                    (
                        Array2::from_elem((4, TRACES_CHUNK_SIZE), [0i16; 8]),
                        Array3::from_elem((4, np, nc as usize), [0i32; 8]),
                    )
                },
                |(traces_tr, tmp_sum), (mut sum, mut sum_square, trace_chunk)| {
                    let mut sample_bits_used_msk = 0;
                    izip!(
                        trace_chunk.axis_chunks_iter(Axis(0), u16::MAX as usize),
                        y.axis_chunks_iter(Axis(1), u16::MAX as usize)
                    )
                    .for_each(|(trace_chunk, y)| {
                        tmp_sum.fill([0; 8]);
                        izip!(
                            trace_chunk.axis_chunks_iter(Axis(0), TRACES_CHUNK_SIZE),
                            y.axis_chunks_iter(Axis(1), TRACES_CHUNK_SIZE)
                        )
                        .for_each(|(trace_chunk, y)| {
                            let mut traces_tr =
                                traces_tr.slice_mut(s![.., ..trace_chunk.shape()[0]]);
                            sample_bits_used_msk |=
                                transpose_traces(traces_tr.view_mut(), trace_chunk);
                            izip!(
                                traces_tr.axis_iter(Axis(0)),
                                tmp_sum.axis_iter_mut(Axis(0)),
                                sum_square.axis_iter_mut(Axis(0)),
                            )
                            .for_each(
                                |(traces_chunk, sum, sum_square)| {
                                    let traces_chunk = traces_chunk.to_slice().unwrap();
                                    // SAFETY: y has been checked before, and offset/stride is in bound
                                    unsafe {
                                        inner_snr_update(
                                            traces_chunk,
                                            y,
                                            sum,
                                            sum_square.into_scalar(),
                                        );
                                    }
                                },
                            );
                        });
                        for (mut sum, tmp_sum) in
                            izip!(sum.axis_iter_mut(Axis(0)), tmp_sum.axis_iter(Axis(0)))
                        {
                            Zip::from(&mut sum).and(tmp_sum).for_each(|sum, tmp_sum| {
                                for (sum, tmp_sum) in sum.iter_mut().zip(tmp_sum.iter()) {
                                    *sum = sum.wrapping_add(&T::tmp2acc(*tmp_sum));
                                }
                            });
                        }
                    });
                    acc_ref.inc(1);
                    sample_bits_used_msk
                },
            )
            .reduce(|| 0, |a, b| a | b);
        self.bit_width = std::cmp::max(self.bit_width, 16 - sample_bits_used_msk.leading_zeros());
        // for any sample x, abs(x) < 2^bit_width
        // we want max_n_samples*abs(x) < T::SumAcc::max_value(), therefore
        // max_n_samples*abs(x) << bit_width \le T::SumAcc::max_value()
        // max_val does not overflow since max_n_samples < 2^32 and self.bit_width < 16
        let max_val = (max_n_samples as i64) << self.bit_width;
        if max_val > T::acc2i64(T::SumAcc::max_value()) {
            return Err(ScalibError::SnrClassOverflow {
                leak_upper_bound: 1 << self.bit_width,
                max_n_traces: max_n_samples as i64,
            });
        }
        return Ok(());
    }

    /// Generate the actual SNR metric based on the current state.
    /// return array axes (variable, samples in trace)
    pub fn get_snr(&self) -> Array2<f64> {
        let mut snr = Array2::<f64>::zeros((self.np, self.ns));
        // on chunks of samples
        (
            self.sum.axis_iter(Axis(0)),
            self.sum_square.axis_iter(Axis(0)),
            snr.axis_chunks_iter_mut(Axis(1), 8),
        )
            .into_par_iter()
            .for_each(|(sum, sum_square, mut snr)| {
                let sum_square: &[i64; 8] = sum_square.into_scalar();
                let general_sum = sum
                    .slice(s![0usize, ..])
                    .iter()
                    .fold([0i64; 8], |mut acc, s| {
                        for (acc, s) in izip!(acc.iter_mut(), s.iter()) {
                            // no overflow: sample on 16 bits, at most 2^32 traces
                            *acc += T::acc2i64(*s);
                        }
                        acc
                    });
                let mut general_sum_sq = [0i128; 8];
                for (sq, s) in izip!(general_sum_sq.iter_mut(), general_sum.iter()) {
                    let s = *s as i128;
                    *sq = s * s;
                }
                // on variables
                izip!(
                    sum.axis_iter(Axis(0)),
                    self.n_samples.axis_iter(Axis(0)),
                    snr.axis_iter_mut(Axis(0))
                )
                .for_each(|(sum, n_samples, snr)| {
                    compute_snr::<T>(
                        sum.to_slice().unwrap(),
                        n_samples.to_slice().unwrap(),
                        sum_square,
                        &general_sum_sq,
                        self.tot_n_samples,
                        snr.into_slice().unwrap(),
                    );
                });
            });
        snr
    }
}

#[inline(never)]
///  # Safety
///  all values in y must be < sum.shape()[1]
unsafe fn inner_snr_update(
    // len: n
    trace_chunk: &[[i16; 8]],
    // (np, n)
    y: ArrayView2<u16>,
    // (np, nc)
    mut sum: ArrayViewMut2<[i32; 8]>,
    sum_square: &mut [i64; 8],
) {
    assert_eq!(trace_chunk.len(), y.shape()[1]);
    assert_eq!(sum.shape()[0], y.shape()[0]);
    for trace in trace_chunk {
        for (sum_square, trace) in sum_square.iter_mut().zip(trace.iter()) {
            let trace = *trace as i64;
            // overflow handled with error elsewhere
            *sum_square = sum_square.wrapping_add(trace * trace);
        }
    }
    izip!(y.outer_iter(), sum.outer_iter_mut()).for_each(|(y, sum)| {
        let sum = sum.into_slice().unwrap();
        izip!(y.to_slice().unwrap(), trace_chunk).for_each(|(y, trace_chunk)| {
            // sum.get_unchecked_mut is safe due to assumption,
            let sum = sum.get_unchecked_mut(*y as usize);
            for j in 0..8 {
                // overflow handled with error elsewhere
                sum[j] = sum[j].wrapping_add(trace_chunk[j] as i32);
            }
        })
    });
}

#[inline(never)]
fn transpose_traces(
    // shape: (4, n)
    mut traces_tr: ArrayViewMut2<[i16; 8]>,
    // shape: (n, ns) with ns <= 32
    trace_chunk: ArrayView2<i16>,
) -> u16 {
    assert_eq!(traces_tr.shape()[1], trace_chunk.shape()[0]);
    assert_eq!(traces_tr.shape()[0], 4);
    assert!(trace_chunk.shape()[1] <= 32);
    let mut max_width: u16 = 0;
    if trace_chunk.shape()[1] == 32 {
        let mut max_width_vec = [0u16; 8];
        izip!(
            traces_tr.axis_iter_mut(Axis(1)),
            trace_chunk.axis_iter(Axis(0))
        )
        .for_each(|(mut traces_tr, trace_chunk)| {
            izip!(
                traces_tr.iter_mut(),
                trace_chunk.axis_chunks_iter(Axis(0), 8)
            )
            .for_each(|(traces_tr, trace_chunk)| {
                let trace_chunk: &[i16; 8] = trace_chunk.to_slice().unwrap().try_into().unwrap();
                //traces_tr.clone_from_slice(trace_chunk);
                *traces_tr = *trace_chunk;
                for (max_width, trace_chunk) in max_width_vec.iter_mut().zip(trace_chunk.iter()) {
                    // i16::abs_diff returns a u16 without overflow nor panic, while i16::abs
                    // panics on i16::min_value() input.
                    *max_width |= trace_chunk.abs_diff(0);
                }
            });
        });
        for mw in max_width_vec {
            max_width |= mw;
        }
    } else {
        izip!(
            traces_tr.axis_iter_mut(Axis(1)),
            trace_chunk.axis_iter(Axis(0))
        )
        .for_each(|(mut traces_tr, trace_chunk)| {
            izip!(
                traces_tr.iter_mut().flat_map(|x| x.iter_mut()),
                trace_chunk.iter()
            )
            .for_each(|(traces_tr, trace_chunk)| {
                *traces_tr = *trace_chunk;
                max_width |= trace_chunk.abs_diff(0);
            });
        });
    }
    return max_width;
}

#[inline(never)]
fn compute_snr<T>(
    sum: &[[T::SumAcc; 8]],
    n_samples: &[u32],
    sum_square: &[i64; 8],
    general_sum_sq: &[i128; 8],
    n: u32,
    snr: &mut [f64],
) where
    T: SnrType,
    T::SumAcc: NativeInt,
{
    let sum_square_class =
        izip!(sum.iter(), n_samples.iter()).fold([0i128; 8], |mut acc, (s, ns)| {
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
