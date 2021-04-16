///! Implementation of Linear Discriminant Analysis templates
///!
///! When the LDA is fit, it computes a linear projection from a high dimensional space
///! to a subspace. The mean in the subspace is estimated for each of the possible nc classes.
///! The covariance of the leakage within the subspace is pooled.
///!
///! Probability estimation for each of the classes based on leakage traces
///! can be derived from the LDA by leveraging the previous templates.
use ndarray::{s, Array1, Array2, Axis, Zip};
use ndarray_linalg::Eigh;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
/// LDA state where leakage has dimension ns. p in the subspace are used.
/// Random variable can be only in range [0,nc[.
pub struct LDA {
    /// Pooled covariance matrix in the subspace. shape (p,p)
    cov: Array2<f64>,
    /// Pseudo inverse of cov. shape (p,p)
    psd: Array2<f64>,
    /// Mean for each of the classes in the subspace. shape (p,nc)
    means: Array2<f64>,
    /// Projection matrix to the subspace. shape of (ns,p)
    projection: Array2<f64>,
    /// Number of dimensions in leakage traces
    ns: usize,
    /// Number of dimensions in the subspace
    p: usize,
    /// Max random variable value.
    nc: usize,
}
#[pymethods]
impl LDA {
    #[new]
    /// Init an LDA with empty arrays
    fn new(_py: Python, nc: usize, p: usize, ns: usize) -> Self {
        LDA {
            cov: Array2::<f64>::zeros((p, p)),
            psd: Array2::<f64>::zeros((p, p)),
            means: Array2::<f64>::zeros((p, nc)),
            projection: Array2::<f64>::zeros((ns, p)),
            nc: nc,
            p: p,
            ns: ns,
        }
    }
    /// Fit the LDA with measurements to derive projection,means,covariance and psd.
    /// x: traces with shape (n,ns)
    /// y: random value realization (n,)
    fn fit(&mut self, py: Python, x: PyReadonlyArray2<i16>, y: PyReadonlyArray1<u16>) {
        let x = x.as_array();
        let y = y.as_array();
        let nc = self.nc;
        let p = self.p;
        let ns = self.ns;
        let n = x.shape()[0];

        // release the GIL
        py.allow_threads(|| {
            // The following goes in three steps
            // 1. Compute the projection
            // 2. Compute the means and cov in linear subspace
            // 3. Compute pseudo inverse of cov

            // ---- Step 1
            // compute the projection
            // This is similar to LDA in scikit-learn with "eigen" parameter
            // ref: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/discriminant_analysis.py#L365

            // sum of the leakage for every possible value
            let mut sums_ns = Array2::<i64>::zeros((nc, ns));
            // number of realization for every value
            let mut s = Array2::<i64>::zeros((nc, 1));

            // compute class sums
            sums_ns
                .outer_iter_mut()
                .into_par_iter()
                .zip(s.outer_iter_mut().into_par_iter())
                .enumerate()
                .for_each(|(i, (mut mean, mut s))| {
                    let mut n = 0;
                    x.outer_iter().zip(y.outer_iter()).for_each(|(x, y)| {
                        let y = y.first().unwrap();
                        if (*y as usize) == i {
                            mean.zip_mut_with(&x, |mean, x| *mean += *x as i64);
                            n += 1;
                        }
                    });
                    s.fill(n);
                });

            let sums_ns = sums_ns.mapv(|x| x as f64);
            let s = s.mapv(|x| x as f64);

            // mean of the traces (without carrying of the classes)
            let mean_total = sums_ns.sum_axis(Axis(0)).insert_axis(Axis(1)) / (n as f64);
            // mean of every class
            let means_ns = sums_ns / s;

            // Compute covariance matrix of x without carrying of the classes store transpose of x
            // in f64.  This consumes twice the memory but allows openblas to be used by ndarray.
            let mut x_f64 = x.mapv(|x| x as f64);
            let mut x_f64_t = Array2::<f64>::zeros(x.t().raw_dim());
            Zip::from(&mut x_f64_t)
                .and(&x_f64.t())
                .par_for_each(|x, y| *x = *y);
            let st = x_f64_t.dot(&x_f64) / (n as f64) - mean_total.dot(&mean_total.t());

            // Computes the covariance of the noise. That is the pooled covariance of the centered
            // traces according to the class. x_f64 = x - means_ns[y]. It uses the same trick with
            // transpose for openblas reversed_axes is to use fortran layout to easily use lapack
            // for the next step
            let mut sw = Array2::<f64>::zeros((ns, ns));
            x_f64
                .outer_iter_mut()
                .into_par_iter()
                .zip(y.outer_iter().into_par_iter())
                .for_each(|(mut x, y)| {
                    let y = y.first().unwrap();
                    x -= &means_ns.slice(s![*y as usize, ..]);
                });
            Zip::from(&mut x_f64_t)
                .and(&x_f64.t())
                .par_for_each(|x, y| *x = *y);
            sw.assign(&(x_f64_t.dot(&x_f64) / (n as f64)));

            // sb = st - sw
            let mut sb = Array2::<f64>::zeros((ns, ns));
            Zip::from(&mut sb)
                .and(&st)
                .and(&sw)
                .par_for_each(|sb, st, sw| *sb = st - sw);

            // Generalized eigenvalue decomposition for sb and sw with a call to dsysgvd Lapack
            // routine. Eigen vectors are stored inplace in sb.
            // See:
            // https://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga912ae48bb1650b2c7174807ffa5456ca.html
            let mut evals = vec![0.0; ns as usize];
            unsafe {
                let itype = 1;
                let i = lapacke::dsygvd(
                    lapacke::Layout::RowMajor,
                    itype,
                    b'V',
                    b'L',
                    ns as i32,
                    sb.as_slice_mut().unwrap(),
                    ns as i32,
                    sw.as_slice_mut().unwrap(),
                    ns as i32,
                    &mut evals,
                );
                if i != 0 {
                    panic!("dsygvd failed, i={}", i);
                }
            }

            // Get the projection from eigen vectors and eigen values
            let mut projection = Array2::<f64>::zeros((p, ns));
            let mut index: Vec<(usize, f64)> = (0..evals.len()).zip(evals).collect();
            let evecs = sb;
            index.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
            index.reverse();
            index
                .iter()
                .zip(projection.axis_iter_mut(Axis(0)))
                .for_each(|((i, _), mut proj)| {
                    proj.assign(&evecs.slice(s![.., *i]));
                });

            // ---- Step 2
            // means per class within the subspace by projecting means_ns
            let means = projection.dot(&means_ns.t());
            // project the centered traces to subspace
            let traces_t = projection.dot(&x_f64_t); // shape (p,n)

            // compute the noise covariance in the linear subspace
            let mut traces_t_t = Array2::zeros(traces_t.t().raw_dim());
            Zip::from(&mut traces_t_t)
                .and(&traces_t.t())
                .par_for_each(|x, y| *x = *y);
            let cov = traces_t.dot(&traces_t_t) / (n as f64);

            // ---- Step 3
            // Compute pseudo inverse of covariance matrix
            // This is a translation of _PSD in scipy.multivariate
            // ref: https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L115

            let (s, u) = cov.eigh(ndarray_linalg::UPLO::Lower).unwrap();
            // threshold for eigen values
            let cond =
                (cov.len() as f64) * (s.fold(f64::MIN, |acc, x| acc.max(f64::abs(*x)))) * 2.3E-16;

            // Only eigen values larger than cond and their id
            // vec<(id,eigenval)>
            let pack: Vec<(usize, &f64)> = (0..s.len())
                .zip(s.into_iter())
                .filter(|(_, &s)| f64::abs(s) > cond)
                .collect();

            // corresponding eigen vectors
            let mut u_s = Array2::<f64>::zeros((u.shape()[0], pack.len()));
            pack.iter()
                .zip(u_s.axis_iter_mut(Axis(1)))
                .for_each(|((i, _), mut u_s)| u_s.assign(&u.slice(s![.., *i])));

            //  psd = eigen_vec * (1/sqrt(eigen_val))
            let psigma_diag: Array1<f64> =
                pack.into_iter().map(|(_, x)| 1.0 / f64::sqrt(*x)).collect();
            let psd = &u_s * &psigma_diag.broadcast(u_s.shape()).unwrap();

            // store the results
            self.cov.assign(&cov);
            self.psd.assign(&psd);
            self.means.assign(&means);
            self.projection.assign(&projection.t());
        })
    }

    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (n,nc). Every row corresponds to one probability distribution
    fn predict_proba<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
    ) -> PyResult<&'py PyArray2<f64>> {
        // This function is based on scipy.mutlivariate.pdf()
        // ref: https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L452

        let x = x.as_array();
        let n = x.shape()[0];
        let means = &self.means;
        let psd = &self.psd;
        let projection = &self.projection;
        let ns = self.ns;
        let p = self.p;
        let nc = self.nc;

        let mut prs = Array2::<f64>::zeros((n, nc));
        py.allow_threads(|| {
            // along with x and prs
            x.outer_iter()
                .into_par_iter()
                .zip(prs.outer_iter_mut().into_par_iter())
                .for_each_init(
                    || {
                        (
                            Array1::<f64>::zeros((ns,)),
                            Array1::<f64>::zeros((p,)),
                            Array1::<f64>::zeros((p,)),
                        )
                    },
                    |(ref mut x_f64, ref mut x_proj, ref mut mu), (x, mut prs)| {
                        x_f64.zip_mut_with(&x, |x, y| *x = *y as f64);
                        // project trace for subspace such that x_proj = projection @ x_i.  we
                        // don't use x_projet = projection.dot(x_i) since ndarray calls openblas
                        // for typical parameters and the concurrent rayon jobs overload the CPUs
                        x_proj.assign(
                            &projection
                                .axis_iter(Axis(1))
                                .map(|p| {
                                    p.iter()
                                        .zip(x_f64.iter())
                                        .fold(0.0, |acc, (p, x_f64)| acc + p * x_f64)
                                })
                                .collect::<Array1<f64>>(),
                        );

                        // prs[c] = exp(-.5*sum(((x_proj-mean[c]) @ psd)**2))
                        prs.assign(
                            &(means
                                .axis_iter(Axis(1))
                                .map(|means| {
                                    mu.assign(&means);
                                    mu.zip_mut_with(&x_proj, |mu, x_proj| *mu -= x_proj);
                                    psd.dot(mu).fold(0.0, |acc, x| acc + x.powi(2))
                                })
                                .collect::<Array1<f64>>()),
                        );
                        prs.mapv_inplace(|x| f64::exp(-0.5 * x));

                        prs /= prs.sum();
                    },
                );
        });
        Ok(&(prs.to_pyarray(py)))
    }

    /// Set the LDA state based on all its parameters
    fn set_state<'py>(
        &mut self,
        _py: Python<'py>,
        cov: PyReadonlyArray2<f64>,
        psd: PyReadonlyArray2<f64>,
        means: PyReadonlyArray2<f64>,
        projection: PyReadonlyArray2<f64>,
        nc: usize,
        p: usize,
        ns: usize,
    ) {
        self.cov.assign(&cov.as_array());
        self.psd.assign(&psd.as_array());
        self.means.assign(&means.as_array());
        self.projection.assign(&projection.as_array());
        self.nc = nc;
        self.p = p;
        self.ns = ns;
    }

    /// Get LDA internal data
    fn get_cov<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.cov.to_pyarray(py))
    }
    fn get_projection<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.projection.to_pyarray(py))
    }
    fn get_means<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.means.to_pyarray(py))
    }
    fn get_psd<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(&self.psd.to_pyarray(py))
    }
}
