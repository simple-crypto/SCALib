//! Estimation for signal-to-noise ratio.
//!
//! An estimation of SNR is represented with a SNR struct. Calling update allows
//! to update the SNR state with fresh measurements. get_snr returns the current value
//! of the estimate.
//! The SNR can be computed for np independent random variables and the same measurements.
//! The measurements are expected to be of length ns. The random variable values must be
//! included in [0,nc[.

use ndarray::{Array2, Array3, Axis, Zip};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
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
#[pymethods]
impl SNR {
    #[new]
    /// Create a new SNR state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// np: number of independent random variable for which SNR must be estimated
    fn new(nc: usize, ns: usize, np: usize) -> Self {
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
    fn update(&mut self, py: Python, traces: PyReadonlyArray2<i16>, y: PyReadonlyArray2<u16>) {
        let x = traces.as_array();
        let y = y.as_array();
        let sum = &mut self.sum;
        let sum_square = &mut self.sum_square;
        let n_samples = &mut self.n_samples;
        // release the GIL
        py.allow_threads(|| {
            // for each of the independent variables
            sum.outer_iter_mut()
                .into_par_iter()
                .zip(sum_square.outer_iter_mut().into_par_iter())
                .zip(n_samples.outer_iter_mut().into_par_iter())
                .zip(y.outer_iter().into_par_iter())
                .for_each(|(((mut sum, mut sum_square), mut n_samples), y)| {
                    // for each of the possible realization of y
                    sum.outer_iter_mut()
                        .into_par_iter()
                        .zip(sum_square.outer_iter_mut().into_par_iter())
                        .zip(n_samples.outer_iter_mut().into_par_iter())
                        .enumerate()
                        .for_each(|(i, ((mut sum, mut sum_square), mut n_samples))| {
                            let mut n = 0;
                            x.outer_iter().zip(y.iter()).for_each(|(x, y)| {
                                // update sum and sum_square if the random value of y is i.
                                if i == *y as usize {
                                    n += 1;
                                    Zip::from(&mut sum).and(&mut sum_square).and(&x).apply(
                                        |sum, sum_square, x| {
                                            let x = *x as i64;
                                            *sum += x;
                                            *sum_square += x.pow(2);
                                        },
                                    );
                                }
                            });
                            n_samples += n;
                        });
                });
        });
    }

    /// Generate the actual SNR metric based on the current state.
    fn get_snr<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let mut snr = Array2::<f64>::zeros((self.np, self.ns));
        let sum = &self.sum;
        let sum_square = &self.sum_square;
        let n_samples = &self.n_samples;

        // for each independent variable
        // Note: not par_iter on the outer loop since the tmp array can be large
        sum.outer_iter()
            .into_par_iter()
            .zip(sum_square.outer_iter())
            .zip(n_samples.outer_iter())
            .zip(snr.outer_iter_mut())
            .for_each(|(((sum, sum_square), n_samples), mut snr)| {
                let mut tmp = Array2::<f64>::zeros(sum.raw_dim());
                // compute mean for each of the classes
                tmp.outer_iter_mut()
                    .into_par_iter()
                    .zip(n_samples.outer_iter().into_par_iter())
                    .zip(sum.outer_iter().into_par_iter())
                    .for_each(|((mut tmp, n_samples), sum)| {
                        let n_samples = *n_samples.first().unwrap() as f64;
                        tmp.zip_mut_with(&sum, |x, y| *x = (*y as f64) / n_samples);
                    });

                // variance of means
                let mean_var = tmp.var_axis(Axis(0), 0.0);

                // compute var for each of the classes. Leverage already compute means.
                tmp.outer_iter_mut()
                    .into_par_iter()
                    .zip(n_samples.outer_iter().into_par_iter())
                    .zip(sum_square.outer_iter().into_par_iter())
                    .for_each(|((mut tmp, n_samples), sum_square)| {
                        let n_samples = *n_samples.first().unwrap() as f64;
                        tmp.zip_mut_with(&sum_square, |x, y| {
                            *x = ((*y as f64) / n_samples) - x.powi(2)
                        });
                    });
                // mean of variance
                let var_mean = tmp.mean_axis(Axis(0)).unwrap();
                snr.assign(&(&mean_var / &var_mean));
            });
        Ok(&(snr.to_pyarray(py)))
    }
}
