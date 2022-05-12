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

use hytra::TrAdder;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use itertools::izip;
use ndarray::{
    s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3,
    Axis,
};
use num_traits::{Bounded, One, PrimInt, Signed, WrappingAdd, Zero};
use rayon::prelude::*;
use std::thread;
use std::time::Duration;

pub trait NativeInt: PrimInt + Signed + WrappingAdd + Send + Sync {}
impl<T: PrimInt + Signed + WrappingAdd + Send + Sync> NativeInt for T {}

pub trait SnrType {
    const UPDATE_SNR_CHUNK_SIZE: usize;
    type SumAcc;
    type Sample;
    fn sample2tmp(s: Self::Sample) -> i32;
    fn sample2i64(s: Self::Sample) -> i64;
    fn tmp2acc(s: i32) -> Self::SumAcc;
    fn acc2i128(acc: Self::SumAcc) -> i128;
}

pub struct SnrType64bit;
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
    fn acc2i128(acc: Self::SumAcc) -> i128 {
        acc as i128
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
    fn acc2i128(acc: Self::SumAcc) -> i128 {
        acc as i128
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
pub struct SNR<T = SnrType64bit>
where
    T: SnrType,
    T::SumAcc: NativeInt,
{
    /// Sum of all the traces corresponding to each of the classes. shape (np,nc,ns)
    sum: Array3<T::SumAcc>,
    /// Sum of squares with shape (ns)
    sum_square: Array1<i64>,
    /// number of samples per class (np,nc)
    n_samples: Array2<u64>,
    /// number of independent variables
    np: usize,
    /// number of samples in a trace
    ns: usize,
    /// max sample bit width
    bit_width: u32,
    /// total number of accumulated traces
    tot_n_samples: u64,
}

/// Size of chunks of trace to handle in a single loop. This should be large enough to limit loop
/// overhead costs, while being small enough to limit memory bandwidth usage by optimizing cache
/// use.
const GET_SNR_CHUNK_SIZE: usize = 1 << 12;

impl<T> SNR<T>
where
    T: SnrType,
    T::SumAcc: NativeInt,
    T::Sample: NativeInt,
{
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which SNR must be estimated
    pub fn new(nc: usize, ns: usize, np: usize) -> Self {
        SNR {
            sum: Array3::<T::SumAcc>::zeros((np, nc, ns)),
            sum_square: Array1::<i64>::zeros(ns),
            n_samples: Array2::<u64>::zeros((np, nc)),
            ns,
            np,
            bit_width: 1,
            tot_n_samples: 0,
        }
    }

    /// Total number of traces accumulated
    fn n_accumulated_traces(&self) -> u64 {
        self.n_samples.slice(s![0, ..]).sum()
    }

    /// Update the SNR state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    pub fn update(&mut self, traces: ArrayView2<T::Sample>, y: ArrayView2<u16>) {
        let x = traces;

        let acc: TrAdder<u64> = TrAdder::new();
        let acc_ref = &acc;

        let n_chunks = (self.ns as f64 / T::UPDATE_SNR_CHUNK_SIZE as f64).ceil() as u64;
        let n_it = n_chunks * self.np as u64 * self.n_samples.shape()[1] as u64;
        let n_updates = x.shape()[0] as u64 * x.shape()[1] as u64 * self.np as u64;

        // Display bar if about 8E9 updates
        if n_updates > (1 << 33) {
            crossbeam_utils::thread::scope(|s| {
                // spawn computing thread
                s.spawn(move |_| {
                    self.update_internal(traces, y, acc_ref);
                });

                // spawn progress bar thread
                s.spawn(move |_| {
                    let pb = ProgressBar::new(n_it);
                    pb.set_style(
                        ProgressStyle::default_spinner()
                            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] (ETA {eta})")
                            .on_finish(ProgressFinish::AndClear),
                    );
                    pb.set_message("Update SNR...");
                    let mut x = 0;
                    while x < n_it {
                        pb.set_position(x);
                        thread::sleep(Duration::from_millis(50));
                        x = acc_ref.get();
                    }
                    pb.finish_and_clear();
                });
            })
            .unwrap();
        } else {
            self.update_internal(traces, y, acc_ref);
        }
    }

    #[inline(never)]
    fn update_internal(
        &mut self,
        traces: ArrayView2<T::Sample>,
        y: ArrayView2<u16>,
        acc_ref: &TrAdder<u64>,
    ) {
        self.tot_n_samples += traces.shape()[0] as u64;
        let mut max_n_samples = 0;
        izip!(self.n_samples.outer_iter_mut(), y.outer_iter()).for_each(|(mut n_samples, y)| {
            y.into_iter().for_each(|y| {
                n_samples[*y as usize] += 1;
                max_n_samples = std::cmp::max(max_n_samples, n_samples[*y as usize]);
            });
        });
        let bit_width = &mut self.bit_width;
        // chunks traces in small parts
        self.bit_width = (
            self.sum
                .axis_chunks_iter_mut(Axis(2), T::UPDATE_SNR_CHUNK_SIZE),
            self.sum_square
                .axis_chunks_iter_mut(Axis(0), T::UPDATE_SNR_CHUNK_SIZE),
            traces.axis_chunks_iter(Axis(1), T::UPDATE_SNR_CHUNK_SIZE),
        )
            .into_par_iter()
            .map(|(sum, sum_square, x)| {
                let mut bit_width: u32 = 0;
                rayon::join(
                    || update_sum_square_block::<T>(sum_square, x, &mut bit_width),
                    || update_sum_block::<T>(sum, x, y, acc_ref),
                );
                bit_width
            })
            .max()
            .max(Some(self.bit_width))
            .unwrap();
        // for any sample x, abs(x) < 2^bit_width
        // we want tot_n_samples*abs(x)^2 \le 2^63, therefore
        // we want tot_n_samples \le 2^{63-2*bit_width) = 1 << 62-2*bit_width
        assert!(
            self.tot_n_samples <= 1u64 << (62 - 2 * self.bit_width),
            "Too many traces: squares accumulator overflow."
        );
        // for any sample x, abs(x) < 2^bit_width
        // we want max_n_samples*abs(x) < T::SumAcc::max_value(), therefore
        // max_n_samples*abs(x) << bit_width \le T::SumAcc::max_value()
        let max_val = max_n_samples << self.bit_width;
        assert!(
            max_val as i128 <= T::acc2i128(T::SumAcc::max_value()),
            "Too many traces: sums accumulator overflow."
        );
    }

    /// Generate the actual SNR metric based on the current state.
    /// return array axes (variable, samples in trace)
    pub fn get_snr(&self) -> Array2<f64> {
        let general_sum = self.sum.slice(s![0usize, .., ..]).sum_axis(Axis(0));
        let mut snr = Array2::<f64>::zeros((self.np, self.ns));
        let n = self.n_accumulated_traces() as i128;
        // For each variable
        izip!(
            self.sum.outer_iter(),
            self.n_samples.outer_iter(),
            snr.outer_iter_mut(),
        )
        .for_each(|(sum, n_samples, mut snr)| {
            let mut sum_square_class = Array1::<i128>::zeros(self.ns);
            // For chunk of samples in trace
            izip!(
                sum.axis_chunks_iter(Axis(1), GET_SNR_CHUNK_SIZE),
                snr.axis_chunks_iter_mut(Axis(0), GET_SNR_CHUNK_SIZE),
                sum_square_class.axis_chunks_iter_mut(Axis(0), GET_SNR_CHUNK_SIZE),
            )
            .for_each(|(sum, mut snr, mut sum_square_class)| {
                // Accumulate square of sums per class
                // For each class
                izip!(sum.outer_iter(), n_samples.iter()).for_each(|(sum, n_samples)| {
                    // For each sample
                    izip!(sum.iter(), sum_square_class.iter_mut()).for_each(
                        |(sum, sum_square_class)| {
                            let sum = T::acc2i128(*sum);
                            let n_traces = *n_samples as i128;
                            *sum_square_class += sum * sum * n / n_traces;
                        },
                    );
                });
                // Compute noise and signal for every sample
                izip!(
                    sum_square_class.iter(),
                    self.sum_square.iter(),
                    general_sum.iter(),
                    snr.iter_mut()
                )
                .for_each(|(sum_square_class, sum_square, general_sum, snr)| {
                    let general_sum = T::acc2i128(*general_sum);
                    let sum_square = *sum_square as i128;
                    let n = n as i128;
                    let signal = sum_square_class - general_sum * general_sum;
                    let noise = n * sum_square - sum_square_class;
                    *snr = (signal as f64) / (noise as f64);
                });
            });
        });
        snr
    }
}

#[inline(never)]
fn inner_loop_update<T: SnrType>(
    sum: ArrayViewMut1<T::SumAcc>,
    x: ArrayView2<T::Sample>,
    y: ArrayView1<u16>,
    i: u16,
    tmp_sum: &mut [i32],
) where
    T::SumAcc: NativeInt,
    T::Sample: NativeInt,
{
    let sum = sum.into_slice().unwrap();
    tmp_sum.fill(0);
    let mut n_accumulated = 0u32;
    izip!(x.outer_iter(), y.iter()).for_each(|(x, v)| {
        if i == *v {
            if n_accumulated >= 1 << 16 && T::SumAcc::count_zeros(T::SumAcc::zero()) > 32 {
                izip!(sum.iter_mut(), tmp_sum.iter()).for_each(|(sum, &tmp_sum)| {
                    *sum = *sum + T::tmp2acc(tmp_sum);
                });
                tmp_sum.fill(0);
                n_accumulated = 0;
            }
            let x = x.to_slice().unwrap();
            izip!(tmp_sum.iter_mut(), x.iter()).for_each(|(sum, &x)| {
                *sum += T::sample2tmp(x);
            });
            n_accumulated += 1;
        }
    });
    izip!(sum.iter_mut(), tmp_sum.iter()).for_each(|(sum, &tmp_sum)| {
        // overflow will be caught later
        *sum = sum.wrapping_add(&T::tmp2acc(tmp_sum));
    });
}

fn update_sum_square_block<T: SnrType>(
    sum_square: ArrayViewMut1<i64>,
    traces: ArrayView2<T::Sample>,
    bit_width: &mut u32,
) where
    T::SumAcc: NativeInt,
    T::Sample: NativeInt,
{
    let mut x_max: T::Sample = T::Sample::one();
    let sum_square = sum_square.into_slice().unwrap();
    for trace in traces.outer_iter() {
        let trace = trace.to_slice().unwrap();
        let mut x_max: T::Sample = T::Sample::one() << (*bit_width as usize - 1);
        izip!(sum_square.iter_mut(), trace).for_each(|(sum_square, &x)| {
            x_max = x_max | x.abs();
            let x = T::sample2i64(x);
            // overflow will be caught later
            *sum_square = sum_square.wrapping_add(x * x);
        });
    }
    *bit_width = 16 - x_max.leading_zeros();
}

fn update_sum_block<T: SnrType>(
    mut sum: ArrayViewMut3<T::SumAcc>,
    traces: ArrayView2<T::Sample>,
    y: ArrayView2<u16>,
    acc_ref: &TrAdder<u64>,
) where
    T::SumAcc: NativeInt,
    T::Sample: NativeInt,
{
    let mut tmp_sum: Vec<i32> = vec![0; traces.shape()[0]];
    // iter on each variable to update
    izip!(sum.outer_iter_mut(), y.outer_iter()).for_each(|(mut sum, y)| {
        // for each of the possible realization of y
        sum.outer_iter_mut().enumerate().for_each(|(i, mut sum)| {
            inner_loop_update::<T>(
                sum.view_mut(),
                traces.view(),
                y.view(),
                i as u16,
                &mut tmp_sum,
            );
            acc_ref.inc(1);
        });
    });
}
