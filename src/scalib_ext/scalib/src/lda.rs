//! Implementation of Linear Discriminant Analysis templates
//!
//! When the LDA is fit, it computes a linear projection from a high dimensional space
//! to a subspace. The mean in the subspace is estimated for each of the possible nc classes.
//! The covariance of the leakage within the subspace is pooled.
//!
//! Probability estimation for each of the classes based on leakage traces
//! can be derived from the LDA by leveraging the previous templates.

#![allow(dead_code)]

use nalgebra::base::*;
use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use rayon::prelude::*;
use std::convert::TryInto;
use std::ops::AddAssign;

/// Accumulator of traces to build LDA
// Compute S_W and S_B as defined in Section 3.8.3, pp. 121-124 of
// R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification (Second Edition). John Wiley & Sons,
// Inc., New York, 2001. ISBN 0-471-05669-3.
pub struct LdaAcc {
    /// Number of classes
    nc: usize,
    /// Number samples per trace
    ns: usize,
    /// Total number of traces
    // u32 is enough since it's the maximum number of squred (almost)
    n: u32,
    /// Non-centered cross-moments of order 2 of the traces (i.e., like a covariance matrix but
    /// don't subtract the mean, and don't divide by the number of traces).  Shape (ns, ns).
    traces_sq: Array2<f64>,
    /// Sum traces for each each class. Shape (nc, ns).
    traces_sum: Array2<f64>,
    /// Number of traces in each class. Shape (nc,).
    // u32 is enough since it's the maximum number of squred (almost)
    n_traces: Array1<u32>,
}
impl LdaAcc {
    fn new(nc: usize, ns: usize) -> Self {
        Self {
            nc,
            ns,
            n: 0,
            traces_sq: Array2::zeros((ns, ns)),
            traces_sum: Array2::zeros((nc, ns)),
            n_traces: Array1::zeros((nc,)),
        }
    }
    /// Traces: shape (n, ns). Classes shape: (n,)
    fn fit_u(&mut self, traces: ArrayView2<i16>, classes: ArrayView1<u16>) {
        assert_eq!(traces.shape()[0], classes.shape()[0]);
        assert_eq!(traces.shape()[1], self.ns);
        self.n = self
            .n
            .checked_add(traces.shape()[0].try_into().unwrap())
            .expect("Too many traces.");
        // Update squares matrix
        // TODO might be worth it to optimize, with a call to gemm...
        let traces = traces.map(|x| *x as f64);
        self.traces_sq
            .zip_mut_with(&traces.t().dot(&traces), |s, t| *s += *t);
        // Update traces sum and number of traces in class
        for (trace, class) in traces.outer_iter().zip(classes.iter()) {
            self.traces_sum
                .slice_mut(s![Into::<usize>::into(*class), ..])
                .add_assign(&trace);
            self.n_traces[Into::<usize>::into(*class)] += 1;
        }
    }
    fn get_matrices(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // Notations:
        // * X be the matrix of data of shape n*ns
        // * 1 be a vector of size n whose all elements are 1.
        // * mu be the row vector of means of shape 1*ns (mu = 1/n*1^T*X).
        // * X_i be all traces of class i, of shape n_i*ns
        // * 1_i be a vector of size n_i whose all elements are 1.
        // * t_i be the row vector of sums of shape 1*ns (t_i = 1_i^T*X_i).
        // * T be the stacking of all t_i.
        // * N be the vector of ni
        // Algo
        // 0. Remark that
        // S_i = (X_i - 1/n_i*1_i*t_i)^T*(X_i - 1/n_i*1_i*t_i)
        //     = X_i^T*X_i - 1/n_i (X_i^T*1_i*t_i + (1_i*t_i)^T*X_i) + 1/n_i^2*t_i^T*1_i^T*1_i*t_i
        //     = X_i^T*X_i - 2/n_i*t_i^T*t_i + 1/n_i*t_i^T*t_i
        //     = X_i^T*X_i - 1/n_i*t_i^T*t_i
        // 1. Compute S_W
        // S_W = \sum_{i=1}^nc S_i
        //     = X^T*X - T^T*(diag(N)^{-1}*T)
        //  Precision ok on 64-bit. overflow: log(n_i)+2*16+log(nc). n ~= n_i*nc -> ok on 64-bit
        //
        // (2. Compute S_T)
        // S_T = (X - 1/n*1*t)^T*(X-1/n*1*t)
        //     = X^T*X - 1/n*t^T*t
        //     = X^T*X - t^T*(t/n)
        //     t is on 16+log(n) bits, hence whole computation holds on 64 bits
        //
        // 3. Compute S_B
        // (S_B = S_T - S_W)
        // S_B = (mus - 1*mu)^T*diag(N)*(mus-1*mu)
        let mus = ndarray::Zip::from(&self.traces_sum)
            .and_broadcast(self.n_traces.slice(s![.., ndarray::NewAxis]))
            .map_collect(|t, n| t / (*n as f64));
        let s_w = &self.traces_sq - self.traces_sum.t().dot(&mus);
        let mut mu = self.traces_sum.sum_axis(ndarray::Axis(0));
        mu.map_inplace(|x| *x /= self.n as f64);
        let c_mus = mus.clone() - mu.slice(s![ndarray::NewAxis, ..]);
        let cmus_scaled = ndarray::Zip::from(&c_mus)
            .and_broadcast(self.n_traces.slice(s![.., ndarray::NewAxis]))
            .map_collect(|m, n| m * (*n as f64));
        let s_b = c_mus.t().dot(&cmus_scaled);
        return (s_w, s_b, mus);
    }
}

pub struct LdaAcc2 {
    /// Number of classes
    nc: usize,
    /// Total number of traces
    // u32 is enough since it's the maximum number of squred (almost)
    n: u32,
    scatter: Array2<f64>,
    /// Sum traces for each each class. Shape (nc, ns).
    traces_sum: Array2<f64>,
    mu: Array1<f64>,
    /// Number of traces in each class. Shape (nc,).
    // u32 is enough since it's the maximum number of squred (almost)
    n_traces: Array1<u32>,
}
impl LdaAcc2 {
    /// Traces: shape (n, ns). Classes shape: (n,)
    fn new(nc: usize, traces: ArrayView2<i16>, classes: ArrayView1<u16>) -> Self {
        let n = traces.shape()[0];
        assert_eq!(n, classes.shape()[0]);
        assert!(n < u32::MAX as usize);
        let traces = traces.map(|x| *x as f64);
        let mut traces_sum: Array2<f64> = Array2::zeros((nc, traces.shape()[1]));
        let mut n_traces: Array1<u32> = Array1::zeros((nc,));
        for (trace, class) in traces.outer_iter().zip(classes.iter()) {
            traces_sum
                .slice_mut(s![Into::<usize>::into(*class), ..])
                .add_assign(&trace);
            n_traces[Into::<usize>::into(*class)] += 1;
        }
        let mu = traces_sum.sum_axis(ndarray::Axis(0)) / (n as f64);
        let traces_c = traces - mu.slice(s![ndarray::NewAxis, ..]);
        let scatter = traces_c.t().dot(&traces_c);
        Self {
            nc,
            n: n as u32,
            scatter,
            traces_sum,
            mu,
            n_traces,
        }
    }
    fn merge(&mut self, other: &Self) {
        assert_eq!(self.nc, other.nc);
        assert_eq!(self.traces_sum.shape()[1], other.traces_sum.shape()[1]);
        let n = self.n.checked_add(other.n).expect("too many traces in LDA");
        let delta_mu = &other.mu - &self.mu;
        self.scatter += &other.scatter;
        // TODO might be worth it to optimize with a call to gemm...
        self.scatter +=
            (self.n as f64) * (other.n as f64) / (n as f64) * delta_mu.t().dot(&delta_mu);
        self.traces_sum += &other.traces_sum;
        self.mu = self.traces_sum.sum_axis(ndarray::Axis(0)) / (n as f64);
        self.n_traces += &other.n_traces;
        self.n = n;
    }
    fn get_matrices(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mus = ndarray::Zip::from(&self.traces_sum)
            .and_broadcast(self.n_traces.slice(s![.., ndarray::NewAxis]))
            .map_collect(|t, n| t / (*n as f64));
        let c_mus = mus.clone() - self.mu.slice(s![ndarray::NewAxis, ..]);
        let cmus_scaled = ndarray::Zip::from(&c_mus)
            .and_broadcast(self.n_traces.slice(s![.., ndarray::NewAxis]))
            .map_collect(|m, n| m * (*n as f64));
        let s_b = c_mus.t().dot(&cmus_scaled);
        let s_w = &self.scatter - &s_b;
        return (s_w, s_b, mus);
    }
}

/// LDA state where leakage has dimension ns. p in the subspace are used.
/// Random variable can be only in range [0,nc[.
pub struct LDA {
    /// Pooled covariance matrix in the subspace. shape (p,p)
    pub cov: Array2<f64>,
    /// Pseudo inverse of cov. shape (p,p)
    pub psd: Array2<f64>,
    /// Mean for each of the classes in the subspace. shape (p,nc)
    pub means: Array2<f64>,
    /// Projection matrix to the subspace. shape of (ns,p)
    pub projection: Array2<f64>,
    /// Number of dimensions in leakage traces
    pub ns: usize,
    /// Number of dimensions in the subspace
    pub p: usize,
    /// Max random variable value.
    pub nc: usize,
}

lazy_static::lazy_static! {
    /// Mutex to avoid calling Lapack from multiple threads simultaneously (admitedly this is quite
    /// hacky).
    static ref LAPACK_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

use geigen::Geigen;

struct LapackGeigen {
    evecs: Array2<f64>,
    evals: Vec<f64>,
}
impl Geigen for LapackGeigen {
    type Error = i32;
    fn new(a: &ArrayView2<f64>, b: &ArrayView2<f64>, n: usize) -> Result<Self, Self::Error> {
        let mut a = a.to_owned();
        let mut b = b.to_owned();
        let ns = a.shape()[0];
        assert_eq!(a.shape()[1], ns);
        assert_eq!(b.shape()[0], ns);
        assert_eq!(b.shape()[1], ns);
        let mut evals = vec![0.0; ns as usize];
        unsafe {
            let guard = LAPACK_MUTEX.lock().unwrap(); // keep this variable until end of use of lapack.
            let itype = 1;
            let i: i32 = lapacke::dsygvd(
                lapacke::Layout::RowMajor,
                itype,
                b'V',
                b'L',
                ns as i32,
                a.as_slice_mut().unwrap(),
                ns as i32,
                b.as_slice_mut().unwrap(),
                ns as i32,
                &mut evals,
            );
            std::mem::drop(guard);
            if i != 0 {
                return Err(i);
            }
        }
        let mut index: Vec<(usize, f64)> = evals.into_iter().enumerate().collect();
        index.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        let mut evecs = Array2::<f64>::zeros((ns, n));
        index
            .iter()
            .zip(evecs.axis_iter_mut(Axis(1)))
            .for_each(|((i, _), mut evec)| {
                evec.assign(&a.slice(s![.., *i]));
            });
        let evals: Vec<f64> = index.into_iter().map(|(_, v)| v).take(n).collect();
        Ok(Self { evecs, evals })
    }
    fn vecs(&self) -> ArrayView2<f64> {
        self.evecs.view()
    }
    fn vals(&self) -> ArrayView1<f64> {
        self.evals.as_slice().into()
    }
}

fn lda_acc_legacy(
    n: usize,
    nc: usize,
    ns: usize,
    x: ArrayView2<i16>,
    y: ArrayView1<u16>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
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
    return (sw, sb, means_ns);
}

impl LDA {
    /// Init an LDA with empty arrays
    pub fn new(nc: usize, p: usize, ns: usize) -> Self {
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
    pub fn fit(
        &mut self,
        x: ArrayView2<i16>,
        y: ArrayView1<u16>,
        eigen_mode: u8,
        cov_mode: u8,
        test_cov: bool,
    ) {
        let nc = self.nc;
        let p = self.p;
        let ns = self.ns;
        let n = x.shape()[0];

        // The following goes in three steps
        // 1. Compute the projection
        // 2. Compute the means and cov in linear subspace
        // 3. Compute pseudo inverse of cov

        // ---- Step 1
        // compute the projection
        // This is similar to LDA in scikit-learn with "eigen" parameter
        // ref: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/discriminant_analysis.py#L365

        if test_cov {
            let (sw, sb, means_ns) = lda_acc_legacy(n, nc, ns, x, y);
            let mut acc = LdaAcc::new(nc, ns);
            acc.fit_u(x, y);
            let (sw2, sb2, means_ns2) = acc.get_matrices();
            let norm_sw2 = &sw2 / (acc.n as f64);
            assert!(
                sw.relative_eq(&norm_sw2, 1e-8, 1e-8),
                "\nsw:\n{:?}\nsw2:\n{:?}\ndiff:\n{:?}",
                sw,
                norm_sw2,
                &sw - &norm_sw2
            );
            assert!(sb.relative_eq(&(sb2 / (acc.n as f64)), 1e-8, 1e-8));
            assert!(means_ns.relative_eq(&means_ns2, 1e-8, 1e-8));

            let acc2 = LdaAcc2::new(nc, x, y);
            let (sw3, sb3, means_ns3) = acc2.get_matrices();
            let norm_sw3 = &sw3 / (acc.n as f64);
            assert!(
                sw.relative_eq(&norm_sw3, 1e-8, 1e-8),
                "\nsw:\n{:?}\nsw2:\n{:?}\ndiff:\n{:?}",
                sw,
                norm_sw2,
                &sw - &norm_sw2
            );
            assert!(sb.relative_eq(&(sb3 / (acc.n as f64)), 1e-8, 1e-8));
            assert!(means_ns.relative_eq(&means_ns3, 1e-8, 1e-8));
            assert!(sw2.relative_eq(&sw3, 1e-6, 1e-6));
        }

        let (sw, sb, means_ns) = match cov_mode {
            0 => lda_acc_legacy(n, nc, ns, x, y),
            1 => {
                let mut acc = LdaAcc::new(nc, ns);
                acc.fit_u(x, y);
                acc.get_matrices()
            }
            2 => LdaAcc2::new(nc, x, y).get_matrices(),
            3 => {
                let mut acc = LdaAcc::new(nc, ns);
                acc.fit_u(x, y);
                let (sw, _, means_ns) = acc.get_matrices();

                let (_, sb, _) = LdaAcc2::new(nc, x, y).get_matrices();
                (sw, sb, means_ns)
            }
            4 => {
                let mut acc = LdaAcc::new(nc, ns);
                acc.fit_u(x, y);
                let (_, sb, means_ns) = acc.get_matrices();

                let (sw, _, _) = LdaAcc2::new(nc, x, y).get_matrices();
                (sw, sb, means_ns)
            }
            _ => unreachable!(),
        };

        // Generalized eigenvalue decomposition for sb and sw with a call to dsysgvd Lapack
        // routine. Eigen vectors are stored inplace in sb.
        // See:
        // https://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga912ae48bb1650b2c7174807ffa5456ca.html
        let projection = match eigen_mode {
            0 => {
                let solver = LapackGeigen::new(&sb.view(), &sw.view(), p).expect("failed to solve");
                solver.vecs().t().to_owned()
            }
            1 => {
                let solver =
                    geigen::GEigenSolver::new(&sb.view(), &sw.view(), p).expect("failed to solve");
                solver.vecs().t().to_owned()
            }
            2 => {
                let solver =
                    geigen::GEigenSolverP::new(&sb.view(), &sw.view(), p).expect("failed to solve");
                solver.vecs().t().to_owned()
            }
            _ => unreachable!(),
        };
        assert_eq!(projection.shape(), [p, ns]);

        // ---- Step 2
        // means per class within the subspace by projecting means_ns
        let means = projection.dot(&means_ns.t());

        // compute the noise covariance in the linear subspace
        // cov= X^T * X
        // proj = (P*X^T)^T = X*P^T
        // cov_proj = (X*P^T)^T*(X*P^T) = P*X^T*X*P^T = P*cov*P^T
        let norm_sw = &sw / (n as f64);
        let cov = projection.dot(&norm_sw.dot(&projection.t()));

        // ---- Step 3
        // Compute pseudo inverse of covariance matrix
        // This is a translation of _PSD in scipy.multivariate
        // ref: https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L115

        let cov_mat = DMatrix::from_iterator(p, p, cov.into_iter().map(|x| *x));
        let eigh = cov_mat.symmetric_eigen();
        let s = eigh.eigenvalues;
        let u = Array::from_iter(eigh.eigenvectors.iter().map(|x| *x)).to_owned();
        let u = u.into_shape((p, p)).unwrap();
        // threshold for eigen values
        let cond = (cov.len() as f64) * (s.fold(f64::MIN, |acc, x| acc.max(f64::abs(x)))) * 2.3E-16;

        // Only eigen values larger than cond and their id
        // vec<(id,eigenval)>
        // FIXME: does this make sense ? The LDA projects into the most significant sub-space...
        let pack: Vec<(usize, &f64)> = (0..s.len())
            .zip(s.into_iter())
            .filter(|(_, &s)| f64::abs(s) > cond)
            .collect();

        // corresponding eigen vectors
        let mut u_s = Array2::<f64>::zeros((p, pack.len()));

        pack.iter()
            .zip(u_s.axis_iter_mut(Axis(1)))
            .for_each(|((i, _), mut u_s)| u_s.assign(&u.slice(s![.., *i])));

        //  psd = eigen_vec * (1/sqrt(eigen_val))
        let psigma_diag: Array1<f64> = pack.into_iter().map(|(_, x)| 1.0 / f64::sqrt(*x)).collect();
        let psd = &u_s * &psigma_diag.broadcast(u_s.shape()).unwrap();

        // store the results
        self.cov.assign(&cov);
        self.psd.assign(&psd);
        self.means.assign(&means);
        self.projection.assign(&projection.t());
    }

    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (n,nc). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, x: ArrayView2<i16>) -> Array2<f64> {
        // This function is based on scipy.mutlivariate.pdf()
        // ref: https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L452

        let n = x.shape()[0];
        let means = &self.means;
        let psd = &self.psd;
        let projection = &self.projection;
        let ns = self.ns;
        let p = self.p;
        let nc = self.nc;

        let mut prs = Array2::<f64>::zeros((n, nc));
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
        return prs;
    }
}
