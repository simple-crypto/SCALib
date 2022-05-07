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
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use rayon::prelude::*;
use std::thread;
use std::time::Duration;
/// SNR state. stores the sum and the sum of squares of the leakage for each of the class.
/// This allows to estimate the mean and the variance for each of the classes which are
/// needed for SNR.
pub struct SNR {
    /// Sum of all the traces corresponding to each of the classes. shape (np,nc,ns)
    sum: Array3<i64>,
    /// Sum of squares with shape (ns)
    sum_square: Array1<i64>,
    /// number of samples per class (np,nc)
    n_samples: Array2<u64>,
    /// number of independent variables
    np: usize,
    /// number of samples in a trace
    ns: usize,
}

/// Size of chunks of trace to handle in a single loop. This should be large enough to limit loop
/// overhead costs, while being small enough to limit memory bandwidth usage by optimizing cache
/// use.
const GET_SNR_CHUNK_SIZE: usize = 1 << 12;
const UPDATE_SNR_CHUNK_SIZE: usize = 1 << 13;

impl SNR {
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which SNR must be estimated
    pub fn new(nc: usize, ns: usize, np: usize) -> Self {
        SNR {
            sum: Array3::<i64>::zeros((np, nc, ns)),
            sum_square: Array1::<i64>::zeros(ns),
            n_samples: Array2::<u64>::zeros((np, nc)),
            ns: ns,
            np: np,
        }
    }

    /// Total number of traces accumulated
    fn n_accumulated_traces(&self) -> u64 {
        self.n_samples.slice(s![0, ..]).sum()
    }

    /// Update the SNR state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView2<u16>) {
        let x = traces;

        if self.n_accumulated_traces() + x.shape()[0] as u64 >= (1 << 32) {
            panic!("SNR can not be updated with more than 2**32 traces.");
        }

        let acc: TrAdder<u64> = TrAdder::new();
        let acc_ref = &acc;

        let n_chunks = (self.ns as f64 / UPDATE_SNR_CHUNK_SIZE as f64).ceil() as u64;
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

    fn update_internal(
        &mut self,
        traces: ArrayView2<i16>,
        y: ArrayView2<u16>,
        acc_ref: &TrAdder<u64>,
    ) {
        let x = traces;
        // chunk the traces to keep one line of sum and sum_square in L2 cache
        (
            self.sum
                .axis_chunks_iter_mut(Axis(2), UPDATE_SNR_CHUNK_SIZE),
            self.sum_square
                .axis_chunks_iter_mut(Axis(0), UPDATE_SNR_CHUNK_SIZE),
            x.axis_chunks_iter(Axis(1), UPDATE_SNR_CHUNK_SIZE),
        )
            .into_par_iter()
            .for_each(|(mut sum, sum_square, x)| {
                // update sum_square
                let sum_square = sum_square.into_slice().unwrap();
                for trace in x.outer_iter() {
                    izip!(sum_square.iter_mut(), trace.to_slice().unwrap()).for_each(
                        |(sum_square, &x)| {
                            let x = x as i64;
                            *sum_square += x * x;
                        },
                    );
                }
                // iter on each variable to update
                (sum.outer_iter_mut(), y.outer_iter())
                    .into_par_iter()
                    .for_each(|(mut sum, y)| {
                        // for each of the possible realization of y
                        sum.outer_iter_mut().into_par_iter().enumerate().for_each(
                            |(i, mut sum)| {
                                inner_loop_update(sum.view_mut(), x.view(), y.view(), i as u16);
                                acc_ref.inc(1);
                            },
                        );
                    });
            });

        // update the number of samples for each classes.
        izip!(self.n_samples.outer_iter_mut(), y.outer_iter()).for_each(|(mut n_samples, y)| {
            y.into_iter().for_each(|y| n_samples[*y as usize] += 1);
        });
    }

    /// Generate the actual SNR metric based on the current state.
    /// return array axes (variable, samples in trace)
    pub fn get_snr<'py>(&self) -> Array2<f64> {
        let general_sum = self.sum.slice(s![0usize, .., ..]).sum_axis(Axis(0));
        let mut snr = Array2::<f64>::zeros((self.np, self.ns));
        let n = self.n_accumulated_traces();
        // For each variable
        (
            self.sum.outer_iter(),
            self.n_samples.outer_iter(),
            snr.outer_iter_mut(),
        )
            .into_par_iter()
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
                                let sum = *sum as i128;
                                let n = n as i128;
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
                    .for_each(
                        |(sum_square_class, sum_square, general_sum, snr)| {
                            let general_sum = *general_sum as i128;
                            let sum_square = *sum_square as i128;
                            let n = n as i128;
                            let signal = sum_square_class - general_sum * general_sum;
                            let noise = n * sum_square - sum_square_class;
                            *snr = (signal as f64) / (noise as f64);
                        },
                    );
                });
            });
        return snr;
    }
}

#[inline(always)]
fn inner_loop_update(sum: ArrayViewMut1<i64>, x: ArrayView2<i16>, y: ArrayView1<u16>, i: u16) {
    let sum = sum.into_slice().unwrap();
    izip!(x.outer_iter(), y.iter()).for_each(|(x, v)| {
        if i == *v {
            let x = x.to_slice().unwrap();
            izip!(sum.iter_mut(), x.iter()).for_each(|(sum, &x)| {
                *sum += x as i64;
            });
        }
    });
}
