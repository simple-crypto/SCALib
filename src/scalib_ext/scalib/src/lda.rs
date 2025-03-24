//! Implementation of Linear Discriminant Analysis templates
//!
//! When the LDA is fit, it computes a linear projection from a high dimensional space
//! to a subspace. The mean in the subspace is estimated for each of the possible nc classes.
//! The covariance of the leakage within the subspace is pooled.
//!
//! Probability estimation for each of the classes based on leakage traces
//! can be derived from the LDA by leveraging the previous templates.
//!
//! ## Algorithm
//! Compute S_W and S_B as defined in Section 3.8.3, pp. 121-124 of
//! R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification (Second Edition). John Wiley & Sons,
//! Inc., New York, 2001. ISBN 0-471-05669-3.
//! Computes S_T and S_B, then computes S_W = S_T - S_B.
//! S_B is computed from class means.
//! S_T is the overall scatter matrix.
//! Algorithms for computation of means and scatter matrix are taken from
//! Pébay, P., Terriberry, T.B., Kolla, H. et al. Numerically stable, scalable formulas for parallel
//! and online computation of higher-order multivariate central moments with arbitrary weights.
//! Comput Stat 31, 1305–1325 (2016). https://doi.org/10.1007/s00180-015-0637-z

use geigen::Geigen;
use ndarray::{s, Array1, Array2, ArrayView2, Axis, NewAxis};
use nshare::{IntoNalgebra, IntoNdarray2};
use serde::{Deserialize, Serialize};

use crate::Result;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// n: total number of traces
    /// p: number of dimensions in reduced space
    /// sw: intra-class covariance
    /// sb: intra-class covariance
    /// means_ns: means per class
    pub fn from_matrices(
        n: usize,
        p: usize,
        sw: ArrayView2<f64>,
        sb: ArrayView2<f64>,
        means_ns: ArrayView2<f64>,
    ) -> Result<Self> {
        let ns = sw.shape()[0];
        let nc = means_ns.shape()[0];
        assert_eq!(ns, sw.shape()[1]);
        assert_eq!(ns, sb.shape()[0]);
        assert_eq!(ns, sb.shape()[1]);
        assert_eq!(ns, means_ns.shape()[1]);
        // LDA here
        // ---- Step 1
        let projection = if p == ns {
            // no LDA, simple pooled gaussian templates
            // This is suboptimal since we then have to pay multiplication with the identity, but
            // it should not matter much (we are probably in a cheap case anyway).
            ndarray::Array2::eye(ns)
        } else {
            // compute the projection
            // Partial generalized eigenvalue decomposition for sb and sw.
            let solver = geigen::GEigenSolverP::new(&sb.view(), &sw.view(), p)?;
            let projection = solver.vecs().into_owned();
            assert_eq!(projection.dim(), (ns, p));
            projection
        };

        // ---- Step 2
        // means per class within the subspace by projecting means_ns
        let means = projection.t().dot(&means_ns.t());
        // compute the noise covariance in the linear subspace
        // cov= X^T * X
        // proj = (P*X^T)^T = X*P^T
        // cov_proj = (X*P^T)^T*(X*P^T) = P*X^T*X*P^T = P*cov*P^T
        let cov = projection.t().dot(&(&sw / (n as f64)).dot(&projection));

        // ---- Step 3
        // Compute the matrix (p, nc) of vectors \omega_k^T
        let cov_mat = cov.view().into_nalgebra();
        let cholesky = cov_mat.cholesky().expect("failed cholesky decomposition");
        let mut omega = means.view().into_nalgebra().into_owned();
        for mut x in omega.column_iter_mut() {
            cholesky.solve_mut(&mut x);
        }
        let omega = omega.into_ndarray2();
        let pk = -0.5 * (&omega * &means).sum_axis(Axis(0));

        Ok(Self {
            projection,
            ns,
            p,
            nc,
            omega,
            pk,
        })
    }

    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// return prs with shape (n,nc). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, x: ArrayView2<i16>) -> Array2<f64> {
        let x = x.mapv(|x| x as f64);
        let mut scores = x.dot(&self.projection).dot(&self.omega) + self.pk.slice(s![NewAxis, ..]);
        for score_distr in scores.outer_iter_mut() {
            softmax(score_distr);
        }
        scores
    }
}
pub(crate) fn softmax(mut v: ndarray::ArrayViewMut1<f64>) {
    let max = v.fold(f64::NEG_INFINITY, |x, y| f64::max(x, *y));
    v.mapv_inplace(|x| f64::exp(x - max));
    let tot = v.sum();
    v /= tot;
}
