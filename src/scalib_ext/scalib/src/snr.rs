//! Estimation for signal-to-noise ratio.
//!
//! An estimation of SNR is represented with a SNR struct. Calling update allows
//! to update the SNR state with fresh measurements. get_snr returns the current value
//! of the estimate.
//! The SNR can be computed for np independent random variables and the same measurements.
//! The measurements are expected to be of length ns. The random variable values must be
//! included in [0,nc[.

use hytra::TrAdder;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Zip};
use rayon::prelude::*;
use std::thread;
use std::time::Duration;
/// SNR state. stores the sum and the sum of squares of the leakage for each of the class.
/// This allows to estimate the mean and the variance for each of the classes which are
/// needed for SNR.
pub struct SNR {
    /// Sum of all the traces corresponding to each of the classes. shape (np,nc,ns)
    sum: Array3<i64>,
    /// sum of squares per class with shape (np,nc,ns)
    sum_square: Array3<i64>,
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
            sum_square: Array3::<i64>::zeros((np, nc, ns)),
            n_samples: Array2::<u64>::zeros((np, nc)),
            ns: ns,
            np: np,
        }
    }

    /// Update the SNR state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (np,n)
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView2<u16>) {
        let x = traces;

        if self.n_samples.slice(s![0, ..]).sum() + x.shape()[0] as u64 >= (1 << 32) {
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
                        thread::sleep(Duration::from_micros(500));
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
                .axis_chunks_iter_mut(Axis(2), UPDATE_SNR_CHUNK_SIZE),
            x.axis_chunks_iter(Axis(1), UPDATE_SNR_CHUNK_SIZE),
        )
            .into_par_iter()
            .for_each(|(mut sum, mut sum_square, x)| {
                // iter on each variable to update
                (
                    sum.outer_iter_mut(),
                    sum_square.outer_iter_mut(),
                    y.outer_iter(),
                )
                    .into_par_iter()
                    .for_each(|(mut sum, mut sum_square, y)| {
                        // for each of the possible realization of y
                        (sum.outer_iter_mut(), sum_square.outer_iter_mut())
                            .into_par_iter()
                            .enumerate()
                            .for_each(|(i, (mut sum, mut sum_square))| {
                                inner_loop_update(
                                    sum.view_mut(),
                                    sum_square.view_mut(),
                                    x.view(),
                                    y.view(),
                                    i as u16,
                                );
                                acc_ref.inc(1);
                            });
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
        // Vor each var, each time sample, compute
        // SNR = Signal / Noise
        // Signal = Var_classes(Mean_traces))
        //     Mean_traces = sum/n_traces
        // Noise = Mean_classes(Var_traces)
        //     Var_traces = 1/n_samples * Sum_traces (trace - Mean_traces)^2
        //                = 1/n_samples * Sum_traces (trace^2 + Mean_traces^2 - 2* trace * Mean_traces)
        //                = Sum_traces (trace^2/n_samples) -  Mean_traces^2
        // TODO check memory layout and algorithmic passes to optimize perf.

        let mut snr = Array2::<f64>::zeros((self.np, self.ns));
        let sum = &self.sum;
        let sum_square = &self.sum_square;
        let n_samples = &self.n_samples;
        let n_samples_inv = 1.0 / n_samples.mapv(|x| x as f64);

        // For each independent variable
        (
            sum.outer_iter(),
            sum_square.outer_iter(),
            n_samples_inv.outer_iter(),
            snr.outer_iter_mut(),
        )
            .into_par_iter()
            .for_each(|(sum, sum_square, n_samples_inv, mut snr)| {
                let mut cum_mean_of_var = Array1::<f64>::zeros(self.ns);
                let mut cum_mean_of_mean = Array1::<f64>::zeros(self.ns);
                let mut cum_var_of_mean = Array1::<f64>::zeros(self.ns);

                izip!(
                    cum_mean_of_var.axis_chunks_iter_mut(Axis(0), GET_SNR_CHUNK_SIZE),
                    cum_mean_of_mean.axis_chunks_iter_mut(Axis(0), GET_SNR_CHUNK_SIZE),
                    cum_var_of_mean.axis_chunks_iter_mut(Axis(0), GET_SNR_CHUNK_SIZE),
                    snr.axis_chunks_iter_mut(Axis(0), GET_SNR_CHUNK_SIZE),
                    sum.axis_chunks_iter(Axis(1), GET_SNR_CHUNK_SIZE),
                    sum_square.axis_chunks_iter(Axis(1), GET_SNR_CHUNK_SIZE)
                )
                .for_each(
                    |(
                        mut cum_mean_of_var,
                        mut cum_mean_of_mean,
                        mut cum_var_of_mean,
                        mut snr,
                        sum,
                        sum_square,
                    )| {
                        // compute mean for each of the classes
                        n_samples_inv
                            .iter()
                            .zip(sum.outer_iter())
                            .zip(sum_square.outer_iter())
                            .enumerate()
                            .for_each(|(i, ((n_samples_inv, sum), sum_square))| {
                                let n_inv = 1.0 / ((i + 1) as f64);
                                Zip::from(&mut cum_mean_of_var)
                                    .and(&mut cum_mean_of_mean)
                                    .and(&mut cum_var_of_mean)
                                    .and(&sum)
                                    .and(&sum_square)
                                    .for_each(
                                        |cum_mean_of_var,
                                         cum_mean_of_mean,
                                         cum_var_of_mean,
                                         sum,
                                         sum_square| {
                                            inner_loop_get_snr(
                                                cum_mean_of_var,
                                                cum_mean_of_mean,
                                                cum_var_of_mean,
                                                sum,
                                                sum_square,
                                                *n_samples_inv,
                                                n_inv,
                                            );
                                        },
                                    );
                            });
                        snr.assign(&(&cum_var_of_mean / &cum_mean_of_var));
                    },
                );
            });
        return snr;
    }
}
/// Incremental update of:
/// - `cum_mean_of_var` and `cum_mean_of_mean`: incremental mean computation
/// - `cum_var_of_mean`: incremental variance computation
fn inner_loop_get_snr(
    cum_mean_of_var: &mut f64,
    cum_mean_of_mean: &mut f64,
    cum_var_of_mean: &mut f64,
    sum: &i64,
    sum_square: &i64,
    n_samples_inv: f64,
    n_inv: f64,
) {
    let u = (*sum as f64) * n_samples_inv;
    let v = (*sum_square as f64) * n_samples_inv - u * u;

    // update the mean of variances estimate
    let v_diff = v - *cum_mean_of_var;
    *cum_mean_of_var += (v_diff) * n_inv;

    // update the variance of means estimate
    let u_diff = u - *cum_mean_of_mean;
    *cum_mean_of_mean += u_diff * n_inv;
    *cum_var_of_mean += ((u_diff * (u - *cum_mean_of_mean)) - *cum_var_of_mean) * n_inv;
}

#[inline(always)]
fn inner_loop_update(
    sum: ArrayViewMut1<i64>,
    sum_square: ArrayViewMut1<i64>,
    x: ArrayView2<i16>,
    y: ArrayView1<u16>,
    i: u16,
) {
    let sum = sum.into_slice().unwrap();
    let sum_square = sum_square.into_slice().unwrap();
    izip!(x.outer_iter(), y.iter()).for_each(|(x, v)| {
        if i == *v {
            let x = x.to_slice().unwrap();
            izip!(sum.iter_mut(), sum_square.iter_mut(), x.iter()).for_each(
                |(sum, sum_square, &x)| {
                    let x = x as i64;
                    *sum += x;
                    *sum_square += x * x;
                },
            );
        }
    });
}
