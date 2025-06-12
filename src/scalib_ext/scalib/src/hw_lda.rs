use crate::ScalibError;

use core::f64;
use geigen::Geigen;
use ndarray::{
    azip, s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, Axis, NewAxis, Zip,
};
use nshare::{IntoNalgebra, IntoNdarray1, IntoNdarray2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::convert::TryInto;
use std::ops::{AddAssign, SubAssign};

/// Hamming Weight LDA: gaussian template model with means of the form A*HW(x)+B
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HwLdaAcc {
    /// Number of variables
    nv: u32,
    /// Number of bits
    nb: u32,
    /// Trace length
    ns: u32,
    /// Number of traces.
    n_traces: u64,
    /// Sum traces Shape (ns,).
    traces_sum: Array1<f64>,
    /// X^T*X (shape nv*2*2), 2 is for linear and affine coefficients
    xtx: Array3<f64>,
    /// X^T*trace (shape nv*2*ns)
    xty: Array3<f64>,
    /// trace^T*trace (ns*ns)
    scatter: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HwLda {
    /// Number of variables
    nv: u32,
    /// Trace length
    ns: u32,
    /// Number of bits
    nb: u32,
    /// Normalized Projection matrix to the subspace. shape of (nv,ns,1)
    norm_proj: Array3<f64>,
    /// Regression coefficients in the projected subspace. Shape of (nv,1,2)
    proj_coefs: Array3<f64>,
}

fn center_hw(hw: u32, nb: u32) -> f64 {
    (hw as f64) - (nb as f64) / 2.0
}
fn centered_hw(class: u64, nb: u32) -> f64 {
    center_hw(class.count_ones(), nb)
}

fn binomials(n: u64) -> Vec<u64> {
    assert!(n > 0);
    if n == 1 {
        return vec![1];
    } else {
        let x = binomials(n - 1);
        let sums = x.iter().zip(x[1..].iter()).map(|(a, b)| a + b);
        std::iter::once(1)
            .chain(sums)
            .chain(std::iter::once(1))
            .collect()
    }
}

impl HwLdaAcc {
    pub fn new(nb: u32, ns: u32, nv: u32) -> Self {
        Self {
            ns,
            nb,
            nv,
            n_traces: 0,
            traces_sum: Array1::zeros((ns as usize,)),
            xtx: Array3::zeros((nv as usize, 2, 2)),
            xty: Array3::zeros((nv as usize, 2, ns as usize)),
            scatter: Array2::zeros((ns as usize, ns as usize)),
        }
    }

    /// Add traces to the accumulator.
    /// traces has shape (nt, ns), classes has shape (nv,nt).
    /// gemm_algo is 0 for ndarray gemm, x>0 for BLIS gemm with x threads.
    pub fn update(&mut self, traces: ArrayView2<i16>, classes: ArrayView2<u64>, gemm_algo: u32) {
        assert_eq!(classes.shape()[0], self.nv as usize);
        assert_eq!(traces.shape()[1], self.ns as usize);
        let nt = traces.shape()[0];
        assert_eq!(classes.shape()[1], nt);

        self.n_traces += u64::try_from(nt).unwrap();
        let traces_buf = traces.mapv(|x| x as f64);
        self.traces_sum.add_assign(&traces_buf.sum_axis(Axis(0)));

        // Accumulate leakage outer product into scatter
        crate::matrixmul::opt_dgemm(
            traces_buf.t(),
            traces_buf.view(),
            self.scatter.view_mut(),
            1.0,
            1.0,
            gemm_algo,
        );

        // Update xtx and xty for each variable
        Zip::indexed(self.xtx.outer_iter_mut())
            .and(self.xty.outer_iter_mut())
            .into_par_iter()
            .for_each(|(k, mut xtx, mut xty)| {
                let classes = classes.slice(s![k, ..]);
                let mut s_hw = 0.0;
                let mut s_hw_sq = 0.0;
                for (c, t) in classes.iter().zip(traces_buf.outer_iter()) {
                    let hw = centered_hw(*c, self.nb);
                    s_hw += hw;
                    s_hw_sq += hw * hw;
                    azip!(xty.slice_mut(s![0, ..]), &t).for_each(|xty, t| *xty += hw * *t);
                }
                *xtx.get_mut((0, 0)).unwrap() += s_hw_sq;
                *xtx.get_mut((1, 0)).unwrap() += s_hw;
                *xtx.get_mut((0, 1)).unwrap() += s_hw;
                *xtx.get_mut((1, 1)).unwrap() += classes.len() as f64;
                xty.slice_mut(s![1, ..]).add_assign(classes.len() as f64);
            });
    }

    fn solve_variable(
        reg_coefs: &mut Array2<f64>, // scratch space
        mut norm_proj: ArrayViewMut2<f64>,
        mut proj_coefs: ArrayViewMut2<f64>,
        xtx: ArrayView2<f64>,
        xty: ArrayView2<f64>,
        scatter: ArrayView2<f64>,
        n: u64,
    ) -> Result<(), ScalibError> {
        // Compute linear regression
        reg_coefs.view_mut().assign(&xty);
        let xtx_nalgebra = xtx.into_nalgebra();

        let cholesky = xtx_nalgebra
            .cholesky()
            .expect("Failed Cholesky decomposition. ");
        cholesky.solve_mut(&mut reg_coefs.view_mut().into_nalgebra());
        // Between class scatter for LDA
        // Original LDA: sb = sum_{traces} (trace-mu)*(trace-mu)^T
        // here, we replace trace with the model coefs^T*b and we get
        //     mu = 1/ntraces * sum_{b} coefs^T*b
        //        = 1/ntraces * coefs^T * sum_{b} b
        //        = 1/ntraces * coefs^T * xtx[0,..] (since b[0] = 1.0 always)
        // Therefore, the scatter is
        //     s_b = sum_{b} (coef^T*b)*(coef^T*b)^T - ntraces*mu*mu^T
        //         = s_m - ntraces*mu*mu^T
        // where we define the model scatter as
        //     s_m = sum_{b} (coef^T*b)*(coef^T*b)^T
        //         = coef^T * [sum_{b} b*b^T] * coef
        //         = coef^T * (self.xtx) * coef
        let nt_mu: Array1<f64> = xtx.slice(s![0usize, ..]).dot(reg_coefs);
        let mu = nt_mu / n as f64;
        let s_m = reg_coefs.t().dot(&xtx).dot(reg_coefs);
        let s_b = &s_m - (n as f64) * mu.slice(s![.., NewAxis,]).dot(&mu.slice(s![NewAxis, ..]));
        // Dimentionality reduction (LDA part)
        // The idea is to solve the generalized eigenproblem (l,w)
        //     s_b*w = l*s_w*w
        // where s_b is the between-classes scatter matrix computed above
        // and s_w is the within-class scatter matrix, in our case it is the scatter of the
        // residual trace-model, where model=coefs^T*b.
        //     sw
        //     = sum_{trace} (trace-coefs^T*b)*(trace-coefs^T*b)^T
        //     = sum_{trace} trace*trace^T - trace*(coefs^T*b)^T - (coefs^T*b)*trace^T  + (coefs^T*b)*(coefs^T*b)^T
        //     = s_t - xty^T*coef - coef.T*xty + s_m
        //     (s_t is self.scatter)
        let s_w = &scatter + s_m - &xty.t().dot(reg_coefs) - &reg_coefs.t().dot(&xty);
        let ns = norm_proj.shape()[1];

        let projection = if ns == 1 {
            Array2::eye(ns)
        } else {
            let solver =
                geigen::GEigenSolverP::new(&s_b.view(), &s_w.view(), 1).expect("failed to solve");
            let projection = solver.vecs().t().into_owned();
            projection
        };
        // Now we can project traces, and projecting the coefs gives us a
        // reduced-dimensionality model.
        // The projection does not guarantee that the scatter of the new residual is unitary
        // (we'd like it to be for later simplicity), hence a apply a rotation.
        // The new residual is projection*(trace-coefs^T*b), hence its scatter is
        // projection*s_w*projection^T
        let cov_proj_res = projection.view().dot(&s_w).dot(&projection.t()) / (n as f64);
        // We decompose cov_proj_res N as N = V*W*V^T where V is orthonormal and W diagonal
        // then if we re-project with W^-1/2*V^T, we get an identity covariance.
        let nalgebra::linalg::SymmetricEigen {
            eigenvectors,
            eigenvalues,
        } = nalgebra::linalg::SymmetricEigen::new(cov_proj_res.into_nalgebra());
        let mut evals = eigenvalues.into_ndarray1();
        let evecs = eigenvectors.into_ndarray2();
        evals.mapv_inplace(|v| 1.0 / v.sqrt());
        let normalizing_proj_t = evecs * evals.slice(s![.., NewAxis]);
        // Storing projections and projected coefficients
        norm_proj.assign(&normalizing_proj_t.t().dot(&projection));
        proj_coefs.assign(&norm_proj.dot(&reg_coefs.t()));
        return Ok(());
    }

    /// Generate projection, projected coefficients, and coefficient chunks
    pub fn solve(&self) -> Result<HwLda, ScalibError> {
        let mut norm_proj = Array3::zeros((self.nv as usize, self.ns as usize, 1));
        let mut proj_coefs = Array3::zeros((self.nv as usize, 1, 2));
        let res = Zip::indexed(norm_proj.outer_iter_mut())
            .and(proj_coefs.outer_iter_mut())
            .into_par_iter()
            .try_for_each_init(
                || return Array2::zeros((2, self.ns as usize)),
                |reg_coefs, (k, norm_proj, proj_coefs)| {
                    Self::solve_variable(
                        reg_coefs,
                        norm_proj,
                        proj_coefs,
                        self.xtx.slice(s![k, .., ..]),
                        self.xty.slice(s![k, .., ..]),
                        self.scatter.view(),
                        self.n_traces,
                    )
                },
            );
        match res {
            Ok(_) => Ok(HwLda {
                nv: self.nv,
                ns: self.ns,
                nb: self.nb,
                norm_proj,
                proj_coefs,
            }),
            Err(err) => Err(err),
        }
    }
}

impl HwLda {
    pub fn project(&self, traces: ArrayView2<i16>, v: u32) -> Array2<f64> {
        traces
            .mapv(|x| x as f64)
            .dot(&self.norm_proj.slice(s![v as usize, .., ..]).t())
    }
    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// v : index of variable that we want to get the probabilities
    /// return prs with shape (n,2**nb). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, traces: ArrayView2<i16>, v: u32) -> Array2<f64> {
        fn softmax(mut v: ndarray::ArrayViewMut1<f64>) {
            v.par_mapv_inplace(|x| f64::exp(x));
            let tot: f64 = Zip::from(v.view()).par_fold(
                || 0.0,
                |acc, s| acc + *s,
                |sum, other_sum| sum + other_sum,
            );
            v.into_par_iter().for_each(|s| *s /= tot);
        }

        let traces = self.project(traces, v);

        // score will contain the squared distance between the trace and the mean of each class
        // it has shape (nt,1<<nb) where nt is the number of traces we need to predict
        // -0.5* || l - A * (Hw(x), 1) || ^ 2
        let mut scores: Array2<f64> = Array2::zeros((traces.len_of(Axis(0)), 1 << self.nb));

        // We force the kernel to allocate pages for scores.
        // This improves speed for large allocations but has no effect on the result
        for t in 0..traces.len_of(Axis(0)) {
            Zip::from(scores.index_axis_mut(Axis(0), t))
                .par_for_each(|x: &mut f64| *x = 0.0 as f64);
        }
        let mut sq_dists = vec![0.0f64; self.nb as usize + 1];

        Zip::from(scores.outer_iter_mut())
            .and(traces.outer_iter())
            .for_each(|mut scores, trace| {
                let trace = trace[0] - self.proj_coefs[(v as usize, 0, 1)];
                let scale = self.proj_coefs[(v as usize, 0, 0)];
                for (hw, sq_dist) in sq_dists.iter_mut().enumerate() {
                    let dist = trace - center_hw(hw as u32, self.nb) * scale;
                    *sq_dist = dist * dist;
                }
                for (class, score) in scores.iter_mut().enumerate() {
                    *score = -0.5 * sq_dists[class.count_ones() as usize];
                }
            });

        for score_distr in scores.outer_iter_mut() {
            softmax(score_distr);
        }
        return scores;
    }

    /// return the log2 probability of one possible value for leakage samples
    /// traces with shape (n,ns)
    /// y with shape (n, nv)
    /// return prs with shape (nv,n), proba of the corresponding y
    pub fn predict_log2p1(&self, traces: ArrayView2<i16>, y: ArrayView2<u64>) -> Array2<f64> {
        let mut proj_traces = Array3::zeros((self.nv as usize, traces.len_of(Axis(0)), 1));
        for (var, mut proj_traces) in proj_traces.outer_iter_mut().enumerate() {
            proj_traces.assign(&self.project(traces, var as u32));
        }

        // score will contain the squared distance between the trace and the mean of each class
        // it has shape (nt,1<<nb) where nt is the number of traces we need to predict
        let mut scores: Array3<f64> = Array3::zeros((
            traces.len_of(Axis(0)),
            self.nv as usize,
            self.nb as usize + 1,
        ));

        Zip::from(scores.outer_iter_mut())
            .and(proj_traces.axis_iter(Axis(1)))
            .for_each(|mut sq_dists, proj_traces| {
                azip!(
                    self.proj_coefs.outer_iter(),
                    sq_dists.outer_iter_mut(),
                    proj_traces.outer_iter()
                )
                .for_each(|proj_coefs, mut sq_dists, proj_trace| {
                    let trace = proj_trace[0] - proj_coefs[(0, 1)];
                    let scale = proj_coefs[(0, 0)];
                    for (hw, sq_dist) in sq_dists.iter_mut().enumerate() {
                        let dist = trace - center_hw(hw as u32, self.nb) * scale;
                        *sq_dist = dist * dist;
                    }
                });
            });

        let scores = scores.mapv(|x| -0.5 * x);

        let bin = binomials(self.nb as u64)
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<_>>();
        let mut res = Array2::zeros(y.dim());
        azip!(res.outer_iter_mut(), scores.outer_iter(), y.outer_iter()).for_each(
            |mut res, scores, y| {
                azip!(res.outer_iter_mut(), scores.outer_iter(), y.outer_iter()).for_each(
                    |res, scores, y| {
                        let max = scores
                            .iter()
                            .fold(f64::NEG_INFINITY, |x, y| f64::max(x, *y));
                        let sum = scores
                            .iter()
                            .zip(bin.iter())
                            .map(|(d, b)| d * f64::exp(b - max))
                            .sum();
                        *res.into_scalar() = (scores[y.into_scalar().count_ones() as usize] - max)
                            * f64::consts::LOG2_E
                            - f64::log2(sum)
                    },
                );
            },
        );
        return res;
    }
}
