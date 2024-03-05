use crate::ScalibError;

use geigen::Geigen;
use kdtree::{distance::squared_euclidean, KdTree};
use ndarray::{
    s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, Axis, NewAxis, Zip,
};
use nshare::{ToNalgebra, ToNdarray1, ToNdarray2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::convert::TryInto;
use std::ops::{AddAssign, SubAssign};

// NBITS_CHUNKS defines the size in bits of the precomputed mean chunks. By default, 8bit chunks are used.
const NBITS_CHUNK: usize = 8;
const SIZE_CHUNK: usize = 1 << NBITS_CHUNK;

/// RLDA contains accumulator, solver and probability predictor.
///
///
/// Implementaion of RLDA
///
/// This module computes the projection matrices and projected regression coeficients of a Regression-based Linear Discriminant analysis.
/// It first calculates a linear regression on nb+1 coefficients (+1 is fot the intercept), then during solve phase, computes regression coefficients in the high-dimensional space, the scatter matrices and the projection to a lower dimensionality matrix.
/// It then applies a second projection to obtain an identity covariance matrix. It applies the 2 projections to the high-dimensional regression coefficients.
///
/// Finally, it precomputes parts of the 2**nb means in chunks, used in the prediction phase.
///
/// In more details,
/// During fitting, each label is mapped to an (nb+1) coefficient array corresponding to the bits of the label + the intercept.
/// A 0 bit is mapped to -1 and a 1 bit is mapped to 1.
/// The intercept is always mapped to 1 and is at the beginning of the array.
/// The matrix X can be defined as the collection of the n coefficient arrays used for profiling and has shape (n,nb+1).
/// In practice it is never computed, instead, for each profiled variable and each trace :
/// the accumulator xtx takes the outer product of the coefficient array with itself,
/// xty takes the outer product of the coefficient array and the leakage,
/// and scatter accumulates the outer product of each trace with itself.
///
/// This precomputation is done in order to avoid computation on all the coefficients for each class for a small increase in state size.
/// The chunks are computed using NBITS_CHUNK coefficients at a time (except for the first one, which includes the intercept).
///
/// We have the mean for a class as follows (for each variable v and projected sample p) :
/// mu[v,class,p] sum_i=0^nb+1 proj_coefs[v,p,i]*int2mul(class,ext_bit(i)
///
/// The sum is splitted into chunks as follows :
/// mu_chunks[v,chunk_id,class,p] = sum_i=0^NBITS_CHUNK proj_coefs[v,NBITS_CHUNK*chunk_id+i,p]*int2mul(class,ext_bit(NBITS_CHUNK*chunk_id+i)
/// The mean is then rewritten as follows :
/// mu[v,class,p] = sum_i=0^n_chunks mu_chunks[v, i, (class>>(NBITS_CHUNK*i))&SIZE_CHUNK ,p]
///
/// During prediction phase, probability estimation is done for all the 2**nb classes based on the leakage traces.
/// Prediction is done by generating for each class the squared distance between the mean of the class and the leakage.
#[derive(Serialize, Deserialize)]
pub struct RLDA {
    /// Number of samples in trace
    pub ns: usize,
    /// Number of bits
    pub nb: usize,
    /// Total number of traces
    pub n: usize,
    /// Number of variables
    pub nv: usize,
    /// Number of dimensions after dimensionality reduction
    pub p: usize,
    /// Sum traces Shape (ns,).
    pub traces_sum: Array1<f64>,
    /// X^T*X (shape nv*(nb+1)*(nb+1)), +1 is for intercept
    pub xtx: Array3<f64>,
    /// X^T*trace (shape nv*(nb+1)*ns), +1 is for intercept
    pub xty: Array3<f64>,
    /// trace^T*trace (ns*ns)
    pub scatter: Array2<f64>,
    /// Normalized Projection matrix to the subspace. shape of (nv,ns,p)
    pub norm_proj: Array3<f64>,
    /// Regression coefficients in the projected subspace. Shape of (nv,p,nb+1)  +1 is for intercept
    pub proj_coefs: Array3<f64>,
    /// Precomputed partial mus. Shape of (nv,n_chunks,size_chunk,p)
    pub mu_chunks: Array4<f64>,
}

/// Coefficient bit from a class. 0th bit is always 1, ith bit is i-1 th bit of c
fn ext_bit(class: u64, i: usize) -> u64 {
    if i == 0 {
        1
    } else {
        (class >> (i - 1)) & 0x1
    }
}

/// Convert a 0/1 bit coefficient into the centered -1.0/1.0 version used for
/// the regression.
fn int2mul(x: u64) -> f64 {
    if x == 0 {
        -1.0
    } else {
        1.0
    }
}

impl RLDA {
    /// Create new RLDA object based on
    /// nb : number of bits of the model
    /// ns : number of high-dimensional leakage samples
    /// nv : number of variables to model
    /// p  : number of dimensions in the reduced space
    pub fn new(nb: usize, ns: usize, nv: usize, p: usize) -> Self {
        let n_chunks: usize = (nb + NBITS_CHUNK - 1) / NBITS_CHUNK;
        Self {
            ns,
            nb,
            nv,
            p,
            n: 0,
            traces_sum: Array1::zeros((ns,)),
            xtx: Array3::zeros((nv, nb + 1, nb + 1)),
            xty: Array3::zeros((nv, nb + 1, ns)),
            scatter: Array2::zeros((ns, ns)),
            norm_proj: Array3::zeros((nv, p, ns)),
            proj_coefs: Array3::zeros((nv, p, nb + 1)),
            mu_chunks: Array4::zeros((nv, n_chunks, SIZE_CHUNK, p)),
        }
    }

    /// Add traces to the accumulator.
    /// traces has shape (nt, ns), classes has shape (nv,nt).
    /// gemm_algo is 0 for ndarray gemm, x>0 for BLIS gemm with x threads.
    pub fn update(&mut self, traces: ArrayView2<i16>, classes: ArrayView2<u64>, gemm_algo: u32) {
        assert_eq!(classes.shape()[0], self.nv);
        assert_eq!(traces.shape()[1], self.ns);
        let nt = traces.shape()[0];
        assert_eq!(classes.shape()[1], nt);

        self.n += nt;
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
                for i in 0..(self.nb + 1) {
                    for j in 0..(self.nb + 1) {
                        let s: f64 = classes
                            .iter()
                            .map(|c| int2mul(ext_bit(*c, i) ^ ext_bit(*c, j)))
                            .sum();
                        *xtx.get_mut((i, j)).unwrap() -= s;
                    }
                }
                for i in 0..(self.nb + 1) {
                    for (c, t) in classes.iter().zip(traces_buf.outer_iter()) {
                        if ext_bit(*c, i) == 0 {
                            xty.slice_mut(s![i, ..]).sub_assign(&t);
                        } else {
                            xty.slice_mut(s![i, ..]).add_assign(&t);
                        }
                    }
                }
            });
    }

    fn solve_variable(
        reg_coefs: &mut Array2<f64>,
        mut norm_proj: ArrayViewMut2<f64>,
        mut proj_coefs: ArrayViewMut2<f64>,
        mut mu_chunks: ArrayViewMut3<f64>,
        xtx: ArrayView2<f64>,
        xty: ArrayView2<f64>,
        scatter: ArrayView2<f64>,
        nb: usize,
        p: usize,
        n: usize,
    ) -> Result<(), ScalibError> {
        //Verify all bits have been profiled -> abs(xtx[0,1:]) != n
        let no_empty_classes = xtx
            .slice(s![0, 1usize..])
            .fold(true, |check, &x| check & (f64::abs(x) != n as f64));
        if !no_empty_classes {
            return Err(ScalibError::EmptyClass);
        }

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

        let projection = if p == ns {
            Array2::eye(ns)
        } else {
            let solver =
                geigen::GEigenSolverP::new(&s_b.view(), &s_w.view(), p).expect("failed to solve");
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

        //Precomputing class values by chunks of defined size
        for data_l in 0..min(SIZE_CHUNK, 1 << nb as usize) {
            for chunk in 0..(nb + NBITS_CHUNK - 1) / NBITS_CHUNK as usize {
                let mut mu_l = mu_chunks.slice_mut(s![chunk, data_l, ..]);
                mu_l.fill(0.0); // Clear chunk in case multiple calls to solve
                if chunk == 0 {
                    for j in 0..p {
                        mu_l[j] += int2mul(ext_bit(data_l as u64, 0)) * proj_coefs[[j, 0]];
                    }
                }
                for i in (chunk * NBITS_CHUNK)..(min(nb, (chunk + 1) * NBITS_CHUNK)) {
                    for j in 0..p {
                        mu_l[j] +=
                            int2mul(ext_bit((data_l << (chunk * NBITS_CHUNK)) as u64, i + 1))
                                * proj_coefs[[j, i + 1]];
                    }
                }
            }
        }
        return Ok(());
    }

    /// Generate projection, projected coefficients, and coefficient chunks
    pub fn solve(&mut self) -> Result<(), ScalibError> {
        let res = Zip::indexed(self.norm_proj.outer_iter_mut())
            .and(self.proj_coefs.outer_iter_mut())
            .and(self.mu_chunks.outer_iter_mut())
            .into_par_iter()
            .try_for_each_init(
                || return Array2::zeros((self.nb + 1, self.ns)),
                |reg_coefs, (k, norm_proj, proj_coefs, mu_chunks)| {
                    RLDA::solve_variable(
                        reg_coefs,
                        norm_proj,
                        proj_coefs,
                        mu_chunks,
                        self.xtx.slice(s![k, .., ..]),
                        self.xty.slice(s![k, .., ..]),
                        self.scatter.view(),
                        self.nb,
                        self.p,
                        self.n,
                    )
                },
            );
        match res {
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }

    /// Generate RLDAClusteredModel object that is used for information estimation
    /// One RLDAClusteredModel is for one variable of RLDA
    ///
    /// var_id : The variable id for which the model is generated
    /// store_associated_classes : if true, it is possible to compute exactly and not estimate the distances for a defined number of clusters
    /// store_marginalized_weights : for future use
    /// max_squared_distance : set the maximum distance between 2 clusters
    /// max_cluster_number : set the maximum number of clusters.
    pub fn get_clustered_model(
        &self,
        var_id: usize,
        store_associated_classes: bool,
        max_squared_distance: f64,
        max_cluster_number: u32,
    ) -> Result<RLDAClusteredModel, ScalibError> {
        return RLDAClusteredModel::initialize(
            self.proj_coefs.slice(s![var_id, .., ..]).into_owned(),
            self.norm_proj.slice(s![var_id, .., ..]).into_owned(),
            self.mu_chunks.slice(s![var_id, .., .., ..]).into_owned(),
            max_squared_distance,
            max_cluster_number,
            store_associated_classes,
        );
    }

    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// v : index of variable that we want to get the probabilities
    /// return prs with shape (n,2**nb). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, x: ArrayView2<i16>, v: usize) -> Array2<f64> {
        // Calculates the exponent of the gaussian templates, in this case, with unit covariance matrix,
        // the squared distance between the mean of the classes and the projected leakage.
        //
        // This method is called in predict_proba, per chunks of SIZE_CHUNK scores to calculate.
        // It calculates and stores the sqdist of classes chunk_index*SIZE_CHUNK to (chunk_index+1)*SIZE_CHUNK-1
        let calculate_sqdist = |chunk_index: usize,
                                mut scores_chunk: ArrayViewMut1<f64>,
                                tmp_mu: &mut Array1<f64>,
                                trace: ArrayView1<f64>,
                                v: usize| {
            // First calculate the most significant chunks (containing MSBs) then process the least signifcant one efficently
            for j in 0..self.p {
                tmp_mu[j] = trace[[j]];
                for chunk in 0..((self.nb + NBITS_CHUNK - 1) / NBITS_CHUNK - 1) as usize {
                    //iterate over chunks except smallest
                    let i_chunk = (chunk_index >> (chunk * NBITS_CHUNK)) & (SIZE_CHUNK - 1);
                    tmp_mu[j] -= self.mu_chunks[[v, chunk + 1, i_chunk, j]];
                }
            }
            // tmp_mu now holds the trace - mu(x) up to the last chunk which differs for each class.
            // if nb<NBITS_CHUNK, iterate over 1<<nb only.
            for i_lsb in 0..min(SIZE_CHUNK, 1 << self.nb) {
                let mut acc: f64;
                for j in 0..self.p {
                    acc = tmp_mu[j] - self.mu_chunks[[v, 0, i_lsb, j]];
                    scores_chunk[[i_lsb]] += -0.5 * acc * acc;
                }
            }
        };

        fn softmax(mut v: ndarray::ArrayViewMut1<f64>) {
            v.par_mapv_inplace(|x| f64::exp(x));
            let tot: f64 = Zip::from(v.view()).par_fold(
                || 0.0,
                |acc, s| acc + *s,
                |sum, other_sum| sum + other_sum,
            );
            v.into_par_iter().for_each(|s| *s /= tot);
        }

        // Project the traces.
        let x = x
            .mapv(|x| x as f64)
            .dot(&self.norm_proj.slice(s![v, .., ..]).t());

        // score will contain the squared distance between the trace and the mean of each class
        // it has shape (nt,1<<nb) where nt is the number of traces we need to predict
        let mut scores: Array2<f64> = Array2::zeros((x.len_of(Axis(0)), 1 << self.nb));

        // We force the kernel to allocate pages for scores.
        // This improves speed for large allocations but has no effect on the result
        for t in 0..x.len_of(Axis(0)) {
            Zip::from(scores.index_axis_mut(Axis(0), t))
                .par_for_each(|x: &mut f64| *x = 0.0 as f64);
        }

        Zip::from(scores.outer_iter_mut())
            .and(x.outer_iter())
            .for_each(|mut scores_trace, trace| {
                // Need to split in 2 cases : if nb<NBITS_CHUNK, then exact_chuns_mut would give an empty iterator
                if self.nb < NBITS_CHUNK {
                    let mut tmp_mu = Array1::<f64>::zeros(self.p);
                    calculate_sqdist(0, scores_trace, &mut tmp_mu, trace, v)
                } else {
                    //Iterate over the chunks
                    Zip::indexed(scores_trace.exact_chunks_mut(SIZE_CHUNK))
                        .into_par_iter()
                        .for_each_init(
                            || return Array1::zeros(self.p),
                            |tmp_mu, (i, scores_chunk)| {
                                calculate_sqdist(i, scores_chunk, tmp_mu, trace, v)
                            },
                        );
                }
            });

        for score_distr in scores.outer_iter_mut() {
            softmax(score_distr);
        }
        return scores;
    }
}

#[derive(Serialize, Deserialize)]
pub struct RLDAClusteredModel {
    /// KdTree datastructure storing centroids and for efficiently finding closest ones.
    pub kdtree: KdTree<f64, usize, Vec<f64>>,
    /// RLDA coefficient matrix: shape(p,nb+1)
    pub coefs: Array2<f64>,
    /// RLDA projection matrix: shape(ns,p)
    pub norm_proj: Array2<f64>,
    /// RLDA mu_chunk matrix: shape(n_chunks,size_chunk,p)
    pub mu_chunks: Array3<f64>,
    /// Vector of the value corresponding to each centroid
    pub centroid_ids: Vec<u64>,
    /// Vector of the weight of a centroid
    pub centroid_weights: Vec<u32>,
    /// Vector of the weight of a centroid if distribution is not uniform
    pub centroid_weights_and: Vec<f64>,
    /// Vector containing every value mapped to a centroid
    pub associated_classes: Option<Vec<Vec<u32>>>,
    /// Maximum distance between each centroid
    pub max_squared_distance: f64,
}

impl RLDAClusteredModel {
    /// Find the nearest centroid to a point in space and get its id and distance from it
    pub fn nearest(&self, point: &[f64]) -> Result<(usize, f64), ScalibError> {
        let res = self.kdtree.nearest(point, 1, &squared_euclidean).unwrap();
        if res.len() == 0 {
            Err(ScalibError::EmptyKdTree)
        } else {
            let (sq_distance, centroid_id) = res[0];
            Ok((*centroid_id, sq_distance))
        }
    }

    /// Perform the clustering
    fn initialize(
        coefs: Array2<f64>,
        norm_proj: Array2<f64>,
        mu_chunks: Array3<f64>,
        max_distance: f64,
        max_cluster_number: u32,
        store_associated_classes: bool,
    ) -> Result<Self, ScalibError> {
        let ndims = coefs.shape()[0];
        let nbits = coefs.shape()[1] as u32;
        let max_squared_distance = max_distance * max_distance;
        let associated_classes = store_associated_classes.then(|| Vec::new());
        let mut clustered_model = RLDAClusteredModel {
            kdtree: KdTree::new(ndims),
            coefs,
            norm_proj,
            mu_chunks,
            centroid_ids: Vec::new(),
            centroid_weights: Vec::new(),
            centroid_weights_and: Vec::new(),
            associated_classes,
            max_squared_distance,
        };
        let mut ncentroids = 0 as u32;
        let mut centroid = vec![0f64; ndims];
        let mut p_hw_and = vec![0f64; nbits as usize];
        for i in 0..nbits {
            p_hw_and[i as usize] = 0.25_f64.powf(i.into()) * 0.75_f64.powf((nbits - 1 - i).into());
        }
        let mut hw: f64;
        for i in 0..2u64.pow(nbits - 1) {
            // calculate mean value for i and its hamming weight
            centroid.fill(0.0);
            hw = 0.0;
            for b in 0..nbits {
                let sign = int2mul(ext_bit(i, b as usize));
                if b > 0 {
                    //Don't do this for intercept
                    hw = hw + (sign + 1.0) / 2.0;
                }
                for (centroid, coef) in centroid
                    .iter_mut()
                    .zip(clustered_model.coefs.slice(s!(.., b as usize)).iter())
                {
                    *centroid += sign * *coef;
                }
            }

            // Get the nearest centroid in the kdtree, if error, the kdtree is empty
            // and need to insert centroid.
            let nearest = clustered_model
                .nearest(centroid.as_slice())
                .unwrap_or((0usize, f64::INFINITY));
            //Insert into the tree if centroid if too far from any centroids in the kdtree
            let c_id = if nearest.1 > clustered_model.max_squared_distance {
                clustered_model
                    .kdtree
                    .add(centroid.clone(), clustered_model.centroid_ids.len())
                    .unwrap();
                clustered_model.centroid_ids.push(i);
                clustered_model.centroid_weights.push(0);
                clustered_model.centroid_weights_and.push(0.0);
                if let Some(acl) = clustered_model.associated_classes.as_mut() {
                    acl.push(Vec::new());
                }
                ncentroids += 1;
                (ncentroids - 1) as usize
            } else {
                nearest.0
            };
            //update model weights and associated classes
            clustered_model.centroid_weights[c_id] += 1;
            clustered_model.centroid_weights_and[c_id] += p_hw_and[hw as usize];
            if let Some(acl) = clustered_model.associated_classes.as_mut() {
                acl[c_id].push(i as u32);
            }
            // Return with error if centroid number limit exceeded
            if ncentroids > max_cluster_number {
                return Err(ScalibError::MaxCentroidNumber);
            }
        }
        return Ok(clustered_model);
    }

    /// Get number if centroids in KdTree
    pub fn get_size(&self) -> u32 {
        self.kdtree.size().try_into().unwrap()
    }

    /// Generate an iterator containing the associated values to the closest centroids.
    pub fn get_close_cluster_centers<'a, 'b, 's>(
        &'s self,
        point: &'a [f64],
        max_popped_classes: usize,
    ) -> Result<impl Iterator<Item = (usize, usize)> + 'b, ScalibError>
    where
        'a: 'b,
        's: 'b,
    {
        let mut n: usize = 0;
        if self.associated_classes.is_none() {
            return Err(ScalibError::NoAssociatedClassesStored);
        } else {
            Ok(self
                .kdtree
                .iter_nearest(point, &squared_euclidean)
                .unwrap()
                .map(|(_d, &c_id)| (c_id, self.associated_classes.as_ref().unwrap()[c_id].len()))
                .take_while(move |(_c_id, n_associated)| {
                    n += n_associated;
                    return n < max_popped_classes;
                }))
        }
    }

    /// Return bounds on the probability of the correct class.
    /// It allows tightnening of the bound by allowing computing exactly
    /// the classes associated to the clusters that contribute the most to the bound.
    /// For this, the model is needed to be initialized with store_associated_centroids=true.
    /// Else, it will only compute the centroids with bounds.
    ///
    /// x : traces with shape (nt,ns)
    /// v : index of variable that we want to get the probabilities
    /// values : array of the correct class for each trace. shape(nt).
    /// max_popped_classes: Number of classes that can be computed exactly.
    /// return bounds on prs with shape (nt).
    pub fn bounded_prs(
        &self,
        x: ArrayView2<i16>,
        values: ArrayView1<u64>,
        max_popped_classes: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let ndims = self.coefs.shape()[0];
        let nbits = self.coefs.shape()[1] - 1;
        let n_chunks: usize = (nbits + NBITS_CHUNK - 1) / NBITS_CHUNK;

        let x = x.mapv(|x| x as f64).dot(&self.norm_proj.t());
        let mut clustered_prs_lower: Array1<f64> = Array1::zeros(x.len_of(Axis(0)));
        let mut clustered_prs_upper: Array1<f64> = Array1::zeros(x.len_of(Axis(0)));

        // Get the exponent, i.e. the squared distance between projected trace and mean of val.
        // Uses precomputed chunks from RLDA.
        let get_exponent = |val: usize, trace: ArrayView1<f64>| {
            let mut exponent = 0.0;
            for d in 0..ndims {
                let mut tmp = trace[[d]];
                for chunk in 0..n_chunks {
                    let i_chunk = (val >> (chunk * NBITS_CHUNK)) & (SIZE_CHUNK - 1);
                    tmp -= self.mu_chunks[[chunk, i_chunk, d]]
                }
                exponent += tmp * tmp;
            }
            return exponent;
        };

        // First, calculate bounds on denominator
        Zip::from(clustered_prs_lower.view_mut())
            .and(clustered_prs_upper.view_mut())
            .and(x.outer_iter())
            .par_for_each(|clustered_denom_lower, clustered_denom_upper, trace| {
                // Get the list of centroids that are close to the leakage and should be computed exactly.
                let close_centroids: Option<Vec<usize>> =
                    self.associated_classes.is_some().then(|| {
                        let mut cl_cc: Vec<usize> = self
                            .get_close_cluster_centers(
                                trace.as_slice().unwrap(),
                                max_popped_classes,
                            )
                            .unwrap()
                            .map(|(c_id, _n_associated)| c_id)
                            .collect();
                        cl_cc.sort();
                        return cl_cc;
                    });

                //Iterate over the centroids, check if the centroids is in close_centroids.
                // If true : Calculate the likelihoods of the associated classes to the centroid.
                // Else : Bound likelihood on centroid and *weight
                (*clustered_denom_upper, *clustered_denom_lower) = Zip::indexed(&self.centroid_ids)
                    .and(&self.centroid_weights)
                    .par_fold(
                        || (0.0, 0.0),
                        |mut denom, c_id, centroid, weight| {
                            if close_centroids
                                .as_ref()
                                .and_then(|cl| cl.binary_search(&c_id).ok())
                                .is_some()
                            {
                                // Iterate over the associated classes to the centroid
                                let exact_denom: f64 = self.associated_classes.as_ref().unwrap()
                                    [c_id]
                                    .iter()
                                    .fold(0.0, |denom, &val| {
                                        denom + f64::exp(-0.5 * get_exponent(val as usize, trace))
                                    });
                                denom.0 += exact_denom;
                                denom.1 += exact_denom;
                            } else {
                                // Bound the centroid
                                let exponent = get_exponent(*centroid as usize, trace);
                                let lower_bound = (f64::sqrt(exponent)
                                    + f64::sqrt(self.max_squared_distance))
                                .powi(2);
                                let upper_bound = (f64::sqrt(exponent)
                                    - f64::sqrt(self.max_squared_distance))
                                .max(0.0)
                                .powi(2);
                                denom.0 += f64::exp(-0.5 * (lower_bound)) * (*weight as f64);
                                denom.1 += f64::exp(-0.5 * (upper_bound)) * (*weight as f64);
                            }
                            return denom;
                        },
                        |sum, other_sum| (sum.0 + other_sum.0, sum.1 + other_sum.1),
                    );
            });

        //Calculate numerator using the correct values
        Zip::from(clustered_prs_lower.view_mut())
            .and(clustered_prs_upper.view_mut())
            .and(values)
            .and(x.outer_iter())
            .into_par_iter()
            .for_each(|(prs_l, prs_u, value, trace)| {
                let pr_num = f64::exp(-0.5 * get_exponent(*value as usize, trace));
                //update prs bounds with numerator (contained denominator previously)
                *prs_l = pr_num / *prs_l;
                *prs_u = pr_num / *prs_u;
            });
        return (clustered_prs_lower, clustered_prs_upper);
    }
}
