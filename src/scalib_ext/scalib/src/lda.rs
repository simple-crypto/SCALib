//! Implementation of Linear Discriminant Analysis templates
//!
//! When the LDA is fit, it computes a linear projection from a high dimensional space
//! to a subspace. The mean in the subspace is estimated for each of the possible nc classes.
//! The covariance of the leakage within the subspace is pooled.
//!
//! Probability estimation for each of the classes based on leakage traces
//! can be derived from the LDA by leveraging the previous templates.

#![allow(dead_code)]

use geigen::Geigen;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, NewAxis};
use nshare::{ToNalgebra, ToNdarray2};
use std::ops::AddAssign;

/// Accumulator of traces to build LDA
///
/// ## Algorithm
/// Compute S_W and S_B as defined in Section 3.8.3, pp. 121-124 of
/// R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification (Second Edition). John Wiley & Sons,
/// Inc., New York, 2001. ISBN 0-471-05669-3.
/// Computes S_T and S_B, then computes S_W = S_T - S_B.
/// S_B is computed from class means.
/// S_T is the overall scatter matrix.
/// Algorithms for computation of means and scatter matrix are taken from
/// Pébay, P., Terriberry, T.B., Kolla, H. et al. Numerically stable, scalable formulas for parallel
/// and online computation of higher-order multivariate central moments with arbitrary weights.
/// Comput Stat 31, 1305–1325 (2016). https://doi.org/10.1007/s00180-015-0637-z
pub struct LdaAcc {
    /// Number of samples in trace
    ns: usize,
    /// Number of classes
    nc: usize,
    /// Total number of traces
    n: usize,
    scatter: Array2<f64>,
    /// Sum traces for each each class. Shape (nc, ns).
    traces_sum: Array2<f64>,
    mu: Array1<f64>,
    /// Number of traces in each class. Shape (nc,).
    n_traces: Array1<usize>,
    traces_buf: Array2<f64>,
    traces_sum_buf: Array2<f64>,
}
impl LdaAcc {
    fn from_dim(nc: usize, ns: usize, n: usize) -> Self {
        Self {
            ns,
            nc,
            n: 0,
            scatter: Array2::zeros((ns, ns)),
            traces_sum: Array2::zeros((nc, ns)),
            mu: Array1::zeros((ns,)),
            n_traces: Array1::zeros((nc,)),
            traces_buf: Array2::zeros((n, ns)),
            traces_sum_buf: Array2::zeros((nc, ns)),
        }
    }
    /// Traces: shape (n, ns). Classes shape: (n,)
    fn new(nc: usize, traces: ArrayView2<i16>, classes: ArrayView1<u16>) -> Self {
        let mut res = Self::from_dim(nc, traces.shape()[1], traces.shape()[0]);
        res.update(traces, classes);
        return res;
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
        self.mu = self.traces_sum.sum_axis(Axis(0)) / (n as f64);
        self.n_traces += &other.n_traces;
        self.n = n;
    }
    fn update(&mut self, traces: ArrayView2<i16>, classes: ArrayView1<u16>) {
        // Number of new traces
        let n = traces.shape()[0];
        assert_eq!(n, classes.shape()[0]);
        assert!(n != 0);
        assert_eq!(traces.shape()[1], self.ns);
        // Get a big enough buffer for centered traces.
        if self.traces_buf.shape()[0] < n {
            self.traces_buf = Array2::zeros((n, self.ns));
        }
        let mut traces_buf: ndarray::ArrayViewMut2<f64> = self.traces_buf.slice_mut(s![0..n, ..]);
        traces_buf.zip_mut_with(&traces, |x, y| *x = *y as f64);
        self.traces_sum_buf.fill(0.0);
        for (trace, class) in traces_buf.outer_iter().zip(classes.iter()) {
            self.traces_sum_buf
                .slice_mut(s![Into::<usize>::into(*class), ..])
                .add_assign(&trace);
            self.n_traces[Into::<usize>::into(*class)] += 1;
        }
        // new traces mean
        let mu: Array1<f64> = self.traces_sum_buf.sum_axis(Axis(0)) / (n as f64);
        // center new traces
        traces_buf -= &mu.slice(s![NewAxis, ..]);

        // new scatter matrix
        //let scatter = traces_buf.t().dot(&traces_buf);
        //self.scatter += &scatter;
        ndarray::linalg::general_mat_mul(1.0, &traces_buf.t(), &traces_buf, 1.0, &mut self.scatter);

        let merged_n = self.n.checked_add(n).expect("too many traces in LDA");
        let delta_mu: Array1<f64> = mu - &self.mu;
        //self.scatter +=
        //    (self.n as f64) * (n as f64) / (merged_n as f64) * delta_mu.t().dot(&delta_mu);
        ndarray::linalg::general_mat_mul(
            (self.n as f64) * (n as f64) / (merged_n as f64),
            &delta_mu.slice(s![.., NewAxis]),
            &delta_mu.slice(s![NewAxis, ..]),
            1.0,
            &mut self.scatter,
        );
        self.traces_sum += &self.traces_sum_buf;
        self.mu = self.traces_sum.sum_axis(Axis(0)) / (merged_n as f64);
        self.n = merged_n;
    }
    fn get_matrices(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mus = ndarray::Zip::from(&self.traces_sum)
            .and_broadcast(self.n_traces.slice(s![.., NewAxis]))
            .map_collect(|t, n| t / (*n as f64));
        let c_mus = mus.clone() - self.mu.slice(s![NewAxis, ..]);
        let cmus_scaled = ndarray::Zip::from(&c_mus)
            .and_broadcast(self.n_traces.slice(s![.., NewAxis]))
            .map_collect(|m, n| m * (*n as f64));
        let s_b = c_mus.t().dot(&cmus_scaled);
        let s_w = &self.scatter - &s_b;
        return (s_w, s_b, mus);
    }
}

/// LDA state where leakage has dimension ns. p in the subspace are used.
/// Random variable can be only in range [0,nc[.
///
/// ## Prediction algorithm
/// Observe that
/// `p(y|k) = (2/pi)^(-d/2) * det(S)^(-1/2) * exp[-1/2*(x-mu_k)^t * S^-1 * (x-mu_k)]`.
/// Let `C = (2/pi)^(-d/2) * det(S)^(-1/2)`, we have
/// ```notrust
/// log p(y|k)/C = -1/2*(x-mu_k)^t * S^-1 * (x-mu_k)
///     = -1/2*(x^t * S^-1 * x - x^t * S^-1 * mu_k - mu_k^t * S^-1 * x + mu_k * S^-1 * mu_k)
///     = mu_k^t * S^-1 * x -1/2* mu_k^T * S^-1 * mu_k + K
///     = omega_k^t*x + P_k + K
/// ```
/// where `omega_k = S^-1 * mu_k` and `P_k = -1/2 * omega_k^t * mu_k`.
/// Therefore, `p(k|y) = softmax( (omega_k^t*x + P_k)_k )`.
///
/// We find `omega_k` by solving `S * omega_k = mu_k` (using Cholesky decomposition of `S`).
///
///  We do not combine multiplication by omega_k and the projection matrix: combined, it would
///  require O(ns*nc) computation (and storage) for the scores, while it is O(ns*p + p*nc) for the
///  split one (W then omega), which is interesting as long as p << nc (which is true, otherwise we
///  could as well take p=nc and not reduce dimensionality).
pub struct LDA {
    /// Projection matrix to the subspace. shape of (ns,p)
    pub projection: Array2<f64>,
    /// Number of dimensions in leakage traces
    pub ns: usize,
    /// Number of dimensions in the subspace
    pub p: usize,
    /// Max random variable value.
    pub nc: usize,
    /// Probability mapping vectors. shape (p,nc)
    pub omega: Array2<f64>,
    /// Probability mapping offsets. shape (nc,)
    pub pk: Array1<f64>,
}

impl LDA {
    /// Init an LDA with empty arrays
    pub fn new(nc: usize, p: usize, ns: usize) -> Self {
        LDA {
            projection: Array2::<f64>::zeros((ns, p)),
            nc: nc,
            p: p,
            ns: ns,
            omega: Array2::<f64>::zeros((p, nc)),
            pk: Array1::<f64>::zeros((nc,)),
        }
    }
    /// Fit the LDA with measurements to derive projection,means,covariance and psd.
    /// x: traces with shape (n,ns)
    /// y: random value realization (n,)
    pub fn fit(&mut self, x: ArrayView2<i16>, y: ArrayView1<u16>) {
        let nc = self.nc;
        let p = self.p;
        let ns = self.ns;
        let n = x.shape()[0];

        // The following goes in three steps
        // 1. Compute the projection
        // 2. Compute the means and cov in linear subspace
        // 3. Compute omega_k and P_k

        // ---- Step 1
        // compute the projection
        // This is similar to LDA in scikit-learn with "eigen" parameter
        // ref: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/discriminant_analysis.py#L365
        let (sw, sb, means_ns) = LdaAcc::new(nc, x, y).get_matrices();

        // Partial generalized eigenvalue decomposition for sb and sw.
        let solver =
            geigen::GEigenSolverP::new(&sb.view(), &sw.view(), p).expect("failed to solve");
        self.projection.assign(&solver.vecs());
        assert_eq!(self.projection.dim(), (ns, p));

        // ---- Step 2
        // means per class within the subspace by projecting means_ns
        let means = self.projection.t().dot(&means_ns.t());
        // compute the noise covariance in the linear subspace
        // cov= X^T * X
        // proj = (P*X^T)^T = X*P^T
        // cov_proj = (X*P^T)^T*(X*P^T) = P*X^T*X*P^T = P*cov*P^T
        let cov = self
            .projection
            .t()
            .dot(&(sw / (n as f64)).dot(&self.projection));

        // ---- Step 3
        // Compute the matrix (p, nc) of vectors \omega_k^T
        let cov_mat = cov.view().into_nalgebra();
        let cholesky = cov_mat.cholesky().expect("failed cholesky decomposition");
        let mut omega = means.view().into_nalgebra().into_owned();
        for mut x in omega.column_iter_mut() {
            cholesky.solve_mut(&mut x);
        }
        self.omega = omega.into_ndarray2();
        self.pk = -0.5 * (&self.omega * &means).sum_axis(Axis(0));
    }

    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (n,nc). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, x: ArrayView2<i16>) -> Array2<f64> {
        fn softmax(mut v: ndarray::ArrayViewMut1<f64>) {
            let max = v.fold(f64::NEG_INFINITY, |x, y| f64::max(x, *y));
            v.mapv_inplace(|x| f64::exp(x - max));
            let tot = v.sum();
            v /= tot;
        }
        let x = x.mapv(|x| x as f64);
        let mut scores = x.dot(&self.projection).dot(&self.omega) + self.pk.slice(s![NewAxis, ..]);
        for score_distr in scores.outer_iter_mut() {
            softmax(score_distr);
        }
        return scores;
    }
}
