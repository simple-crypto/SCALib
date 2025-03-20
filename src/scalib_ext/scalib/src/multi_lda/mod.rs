mod poi_map;
mod scatter_pairs;
mod sparse_trace_sums;
mod utils;

use std::cmp;
use std::ops::Range;
use std::sync::Arc;

use hytra::TrAdder;
use itertools::{izip, Itertools};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::lda::{softmax, LDA};
use crate::{Result, ScalibError};
pub use poi_map::{PoiMap, AA};
pub use scatter_pairs::ScatterPairs;
pub use sparse_trace_sums::SparseTraceSums;
use utils::{log2_softmax_i, ArrayBaseExt, RangeExt};

pub type Class = u16;
pub type Var = u16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLdaAcc {
    /// Number of variables
    nv: Var,
    /// Number of classes
    nc: Class,
    /// Trace length
    ns: u32,
    /// Pois for each var
    cov_pois_offsets: Vec<usize>,
    pub poi_map: Arc<PoiMap>,
    /// Number of traces.
    n_traces: u32,
    /// Per-class sums.
    pub trace_sums: SparseTraceSums,
    /// Trace global scatter.
    pub cov_pois: ScatterPairs,
}

impl MultiLdaAcc {
    pub fn new(ns: u32, nc: Class, mut pois: Vec<Vec<u32>>) -> Result<Self> {
        // Sort POIs: required for SparseTraceSums and has not impact on the LDA result.
        for pois in pois.iter_mut() {
            pois.sort_unstable();
        }
        let nv: Var = pois
            .len()
            .try_into()
            .map_err(|_| ScalibError::TooManyPois)?;
        let poi_map = Arc::new(PoiMap::new(ns as usize, &pois)?);
        let trace_sums = SparseTraceSums::new(ns, nv, nc, poi_map.clone());
        let mapped_pairs = (0..nv).flat_map(|v| poi_map.mapped_pairs(v));
        let cov_pois_offsets = pois
            .iter()
            .scan(0, |acc, x| {
                let res = *acc;
                *acc += Self::npairs_n(x.len());
                Some(res)
            })
            .collect_vec();

        let cov_pois = ScatterPairs::new(poi_map.len(), mapped_pairs)?;
        Ok(Self {
            nv,
            nc,
            ns,
            cov_pois_offsets,
            poi_map,
            n_traces: 0,
            trace_sums,
            cov_pois,
        })
    }

    /// traces shape: (ntraces, ns)
    /// y shape: (ntraces, nv)
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView2<Class>) -> Result<()> {
        assert_eq!(traces.shape()[0], y.shape()[0]);
        assert_eq!(traces.shape()[1], self.ns as usize);
        assert_eq!(y.shape()[1], self.nv as usize);
        self.n_traces = self
            .n_traces
            .checked_add(
                traces.shape()[0]
                    .try_into()
                    .map_err(|_| ScalibError::TooManyVars)?,
            )
            .ok_or(ScalibError::TooManyTraces)?;
        self.trace_sums.update(traces, y);
        self.cov_pois.update(&self.poi_map, traces)?;
        Ok(())
    }

    pub fn ntraces(&self) -> u32 {
        self.n_traces
    }

    /// sums shape: npoi*nc
    fn s_b_mat(&self, sums: &Array2<i64>, n_traces: ArrayView1<u32>) -> Array2<f64> {
        // Between-class scatter matrix computation
        // x == data, xi == data with class i
        // n == number of traces, ni == number of traces with class i
        // Sx(.) == sum over all traces
        // Sxi(.) == sum over traces of class i
        // mu = Sx(x)/n (average)
        // mui = Sxi(xi)/ni
        // ^T= transpose operator
        //
        // Classical computation performed as
        // s_b = 1/n Sum_i[ ni*(mui - mu)(mui - mu)^T]
        //
        // Simplified formulation (used here, constant 1/n factor omitted)
        // s_b = Sum_i[(1/ni)*(Sxi - ni*Sx/n)(Sxi- ni*Sx/n)]
        //
        // Compute the total sum if trace as a matrix of shape (npois, 1);
        let inv_n = 1.0 / (n_traces.sum() as f64);
        let mu = sums.sum_axis(Axis(1)).mapv(|e| (e as f64) * inv_n);
        let n_traces_f = n_traces.mapv(|x| x as f64);
        let inv_n_traces_sqrt = n_traces_f.mapv(|x| 1.0 / x.sqrt());
        // (Sxi - ni*Sx/n)/sqrt(ni)
        let c_sxi_sqrtni = Zip::from(sums)
            .and_broadcast(&mu.insert_axis(Axis(1)))
            .and_broadcast(&n_traces_f.insert_axis(Axis(0)))
            .and_broadcast(&inv_n_traces_sqrt.insert_axis(Axis(0)))
            .map_collect(|sum, mu, ni, inv_ni_sqrt| ((*sum as f64) - *ni * *mu) * *inv_ni_sqrt);
        // Compute the final s_b matrix, by computing the squaring
        // as the result of the dot product between the matric and its transpose.
        c_sxi_sqrtni.dot(&(c_sxi_sqrtni.t()))
    }

    fn compute_matrices_var(&self, var: Var) -> Result<LdaMatrices> {
        // LDA matrices computation.
        // x == data, xi == data with class i
        // n == number of traces, ni == number of traces with class i
        // Sx(.) == sum over all traces
        // Sxi(.) == sum over traces of class i
        // mu = Sx(x)/n (average)
        //
        // # Classic LDA definitions:
        // - between-class scatter: s_b = Si(ni*cmui**2)
        // - within-class scatter: s_w = Si(Sxi((xi-mui)**2))
        // - total scatter: s_t = Sx((x-mu)**2)
        //
        // - between-class scatter
        //    s_b = Si((Sxi(xi)-ni*mu)**2/ni)
        //        = Si((P)**2/ni)
        //    P = Sxi(xi)-(Sx(x)*ni/n)
        // - s_t = s_t_u - Sx(x)**2/n
        //      - s_t_u computed out of CovAcc, over 64 bits
        //          -> (signed -> 2**15)**2 * n -> 62-bit
        //      s_t = (s_t_u*n - Sx(x)**2) as f64 / n as f64
        // - s_w = s_t - s_b

        // shape (n_pois, nc)
        let sums = self.trace_sums.sums_var(var);
        // shape (nc, )
        let n_traces = self.trace_sums.ntraces_var(var);
        if n_traces.iter().any(|n| *n == 0) {
            return Err(ScalibError::EmptyClass);
        }
        // Compute the total sum of traces with shape (npois, )
        let s_x = sums.sum_axis(Axis(1));
        // Total amount of traces used
        let n = n_traces.sum();
        //// Between-class scatter computation.
        let s_b = self.s_b_mat(&sums, n_traces);
        ///// Within-class scatter computation.
        let mut s_w = Array2::zeros((sums.shape()[0], sums.shape()[0]));
        let inv_n = 1.0 / (n as f64);
        for (i, j) in self.var_pairs(var) {
            let (i, j) = (i as usize, j as usize);
            let i2 = self.poi_map.new_pois(var)[i];
            let j2 = self.poi_map.new_pois(var)[j];
            // Total scatter offset by mu**2*n_tot.
            let s_t_u_e = self.cov_pois.get_scatter(i2, j2);
            // Computation of (s_t_u*n - Sx(x)**2) for the target POI pair.
            let t = (s_t_u_e as i128) * (n as i128) - (s_x[i] as i128) * (s_x[j] as i128);
            // Computation of the Within-class element relative to the target POI pair.
            // in practice, performs the operation s_t - s_b, and populate the s_w matrix
            // at proper location.
            let s_w_e = ((t as f64) * inv_n) - s_b[(i, j)];
            s_w[(i, j)] = s_w_e;
            s_w[(j, i)] = s_w_e;
        }
        let inv_ni = n_traces.mapv(|x| 1.0 / (x as f64));
        let mus = Zip::from(sums.t())
            .and_broadcast(&inv_ni.insert_axis(Axis(1)))
            .map_collect(|s, inv_ni| (*s as f64) * inv_ni);
        Ok(LdaMatrices {
            s_w,
            s_b,
            mus,
            n_traces: self.n_traces,
        })
    }
    pub fn get_matrices(&self) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>> {
        Ok((0..self.nv)
            .map(|var| Ok(LdaMatrices::to_tuple(self.compute_matrices_var(var)?)))
            .collect::<Result<_>>()?)
    }
    pub fn lda(&self, p: u32, config: &crate::Config) -> Result<MultiLda> {
        let compute_ldas = |it_cnt: &TrAdder<u64>| {
            (0..self.nv)
                .into_par_iter()
                .map(|var| {
                    let matrices = self.compute_matrices_var(var)?;
                    let res = matrices.lda(p)?;
                    it_cnt.inc(1);
                    Ok(res)
                })
                .collect::<Result<_>>()
        };
        let ldas =
            crate::utils::with_progress(compute_ldas, self.nv as u64, "LDA solve vars", config)?;
        Ok(MultiLda::new(self.poi_map.clone(), self.nc, ldas, p))
    }
    fn pairs_n(n: u32) -> impl Iterator<Item = (u32, u32)> {
        (0..n).flat_map(move |i| (i..n).map(move |j| (i, j)))
    }
    fn var_pairs(&self, var: Var) -> impl Iterator<Item = (u32, u32)> {
        Self::pairs_n(self.poi_map.n_pois(var) as u32)
    }
    fn npairs_n(n: usize) -> usize {
        n * (n + 1) / 2
    }
}

/// Matrices for solving an LDA
#[derive(Debug, Clone)]
pub struct LdaMatrices {
    /// Between-class scatter
    pub s_b: Array2<f64>,
    /// Within-class scatter
    pub s_w: Array2<f64>,
    /// Class means
    pub mus: Array2<f64>,
    /// Number of traces
    pub n_traces: u32,
}

impl LdaMatrices {
    fn to_tuple(self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        (self.s_w, self.s_b, self.mus)
    }
    /// Solve the LDA.
    fn lda(&self, p: u32) -> Result<LDA> {
        LDA::from_matrices(
            self.n_traces as usize,
            p as usize,
            self.s_w.view(),
            self.s_b.view(),
            self.mus.view(),
        )
    }
}

// MultiLda.predict_proba algorithm:
//
// Iteration dimentions:
// - variables
// - p (projected dimensions)
// - samples in trace
// - traces
//
// The core loop is made of FMA with one accumulator for N traces being updated
// across the samples in the trace. That core loop body loads one f64 projection
// coefficient and N i16 samples.
// Therefore, we need N >= 4 for decent AVX2, N >= 32 good AVX2 at 4 cycles
// latency and 0.5 CPI. For full cache-line usage, we also need N >= 32.
//
// Then we iterate over a block of POIs of a var, having the trace in L2 cache.
// Trace L2 cache usage: tot_pois*N*2B, 8B/FMA -> if 2 FMA/cycle,
// use 16B/cycle bandwidth.
//
// Then we iterate over all projection dimensions p. We use exactly the same traces, but
// different projection coefficients and different intermediate results.
//
// Then we iterate over a block of variables. We keep projection coefficients and
// intermediate results in L3 cache.
//
// Then we iterate over the POI blocks.
//
// L3 cache usage:
// - projection coefficients: var_block_size*p*ns_per_var*8B
// - intermediate results: var_block_size*p*N*8B
//
// L3 bandwidth (assuming AVX2: FMA_VEC_SIZE=4)
// - projection coefficients: 8B/(N/FMA_VEC_SIZE)/inst = 32/N B/inst
// - intermediate results: 8*FMA_VEC_SIZE/poi_block_size = 32/poi_block_size B/inst
//
// Then, we iterate over all trace blocks (in parallel over cores).
// Finally, we iterate over varaible blocks.
//
// Constraints
// N = 32
// poi_block_size*N*2B < L2/2
// var_block_size*p*ns_per_var*8B < L3_CCX/2 (L3_CCX = L3 aggregated per core complex/die/...)
// var_block_size*p*N*8B < L3/ncores/4 (unlikely to be limiting)
//
// Now, let us make some conservative assuptions regarding cache sizes
// - L2 = 512 kB
// - L3_CCX = 8 MB
// - L3/ncores = 2 MB
const N: usize = 32;
const L2_SIZE: usize = 512 * 1024;
const L3_CCX: usize = 8 * 1024 * 1024;
const L3_CORE: usize = 2 * 1024 * 1024;
const POI_BLOCK_SIZE: usize = L2_SIZE / 2 / 2 / N;

/// Solved LDA for multiple variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLda {
    /// Number of classes
    nc: Class,
    /// Post-projection dimensions
    p: usize,
    poi_map: Arc<PoiMap>,
    /// poi_blocks[var][poi_block].0: indices of the block's POIs within all POIs of that var.
    /// poi_blocks[var][poi_block].1: indices of POIs relative to the block offset (i*POI_BLOCK_SIZE)
    poi_blocks: Vec<Vec<(Range<usize>, Vec<u16>)>>,
    /// Solved LDA for each var.
    lda_states: Vec<Arc<LdaState>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LdaState {
    // projection and omega are transposed compared to what is found on LDA.
    // shape (p, ns)
    projection: Array2<f64>,
    // shape (nc, p)
    omega: Array2<f64>,
    pk: Array1<f64>,
}

impl MultiLda {
    fn new(poi_map: Arc<PoiMap>, nc: Class, ldas: Vec<LDA>, p: u32) -> Self {
        let poi_blocks = poi_map.poi_blocks();
        let p = p as usize;
        let lda_states = ldas
            .iter()
            .map(|lda| {
                Arc::new(LdaState {
                    projection: lda.projection.t().clone_row_major(),
                    omega: lda.omega.t().clone_row_major(),
                    pk: lda.pk.clone(),
                })
            })
            .collect();
        Self {
            nc,
            p,
            poi_map: poi_map.clone(),
            poi_blocks,
            lda_states,
        }
    }

    /// Number of variables per block.
    fn var_block_size(&self) -> usize {
        let max_pois_per_var = self
            .poi_map
            .new_pois_vars()
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(1);
        let var_block_size = std::cmp::min(
            L3_CCX / 2 / 8 / max_pois_per_var / self.p,
            L3_CORE / 4 / 8 / self.p,
        );
        if var_block_size == 0 {
            eprintln!("WARNING: MultiLda.predict_proba not optimized for these parameters");
            1
        } else {
            var_block_size
        }
    }

    fn nv(&self) -> usize {
        self.poi_blocks.len()
    }

    /// return the probability of each of the possible value for leakage samples
    /// traces with shape (n,ns)
    /// return prs with shape (nv,n,nc). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, traces: ArrayView2<i16>) -> Array3<f64> {
        let mut res = Array3::zeros((self.nv(), traces.shape()[0], self.nc as usize));
        let traces_batched = self.poi_map.select_batches::<N>(traces);
        for var_block in (0..(self.nv())).range_chunks(self.var_block_size()) {
            let p_traces = self.project_traces(var_block.clone(), &traces_batched);
            for (var_i, var) in var_block.clone().enumerate() {
                let mut res = res.index_axis_mut(Axis(0), var);
                (res.axis_chunks_iter_mut(Axis(0), N), p_traces.as_slice())
                    .into_par_iter()
                    .for_each(|(mut res, p_traces)| {
                        let p_traces = p_traces.index_axis(Axis(0), var_i);
                        self.compute_ll_thread_loop(
                            var,
                            res.view_mut(),
                            p_traces.as_slice().unwrap(),
                        );
                        // TODO: improve the softmax.
                        for res in res.outer_iter_mut() {
                            softmax(res);
                        }
                    });
            }
        }
        res
    }

    /// return the log2 probability of one possible value for leakage samples
    /// traces with shape (n,ns)
    /// y with shape (n, nv)
    /// return prs with shape (nv,n), proba of the corresponding y
    pub fn predict_log2p1(&self, traces: ArrayView2<i16>, y: ArrayView2<Class>) -> Array2<f64> {
        let mut res = Array2::zeros((self.poi_blocks.len(), traces.shape()[0]));
        let traces_batched = self.poi_map.select_batches::<N>(traces);
        for var_block in (0..(self.poi_blocks.len())).range_chunks(self.var_block_size()) {
            let p_traces = self.project_traces(var_block.clone(), &traces_batched);
            for (var_i, var) in var_block.clone().enumerate() {
                let y = y.index_axis(Axis(1), var);
                let mut res = res.index_axis_mut(Axis(0), var);
                (
                    res.axis_chunks_iter_mut(Axis(0), N),
                    p_traces.as_slice(),
                    y.axis_chunks_iter(Axis(0), N),
                )
                    .into_par_iter()
                    .for_each_init(
                        || Array2::zeros((N, self.nc as usize)),
                        |tmp_ll, (res, scores, y)| {
                            let scores = scores.index_axis(Axis(0), var_i);
                            self.compute_ll_thread_loop(
                                var,
                                tmp_ll.view_mut(),
                                scores.as_slice().unwrap(),
                            );
                            // TODO: improve the softmax.
                            for (res, ll, y) in izip!(res, tmp_ll.outer_iter(), y) {
                                *res = log2_softmax_i(ll, *y as usize);
                            }
                        },
                    );
            }
        }
        res
    }

    /// Project traces for all vars in a block.
    /// traces_batched: see PoiMap::select_batches
    /// returns: for each trace batch, an array of shape (len of var_block, p) containing the
    /// projected traces.
    fn project_traces(
        &self,
        var_block: Range<usize>,
        traces_batched: &Array2<AA<N>>,
    ) -> Vec<Array2<[f64; N]>> {
        let mut p_traces = vec![
            Array2::from_elem((var_block.len(), self.p), [0.0f64; N]);
            traces_batched.shape()[0]
        ];
        (traces_batched.outer_iter(), p_traces.as_mut_slice())
            .into_par_iter()
            .for_each(|(trace_batch, p_traces)| {
                self.project_thread_loop(
                    var_block.clone(),
                    p_traces,
                    trace_batch.as_slice().unwrap(),
                );
            });
        p_traces
    }

    /// Project traces for all vars in a block and for one batch of traces.
    /// traces_batch length: ns.
    /// updates p_traces, of shape (len of var_block, p)
    fn project_thread_loop(
        &self,
        var_block: Range<usize>,
        p_traces: &mut Array2<[f64; N]>,
        trace_batch: &[AA<N>],
    ) {
        for (poi_block, poi_block_range) in self.poi_block_ranges().enumerate() {
            let trace_batch = &trace_batch[poi_block_range.clone()];
            for (var, mut p_trace_chunk) in var_block.clone().zip(p_traces.outer_iter_mut()) {
                let pois_var_range = &self.poi_blocks[var][poi_block].0;
                let pois = self.poi_blocks[var][poi_block].1.as_slice();
                let proj = &self.lda_states[var].projection;
                for (p_sample_chunk, coefs) in p_trace_chunk.iter_mut().zip(proj.outer_iter()) {
                    let coefs = &coefs.as_slice().unwrap()[pois_var_range.clone()];
                    self.project_inner_loop(p_sample_chunk, trace_batch, coefs, pois);
                }
            }
        }
    }

    /// increment acc with the inner product of proj_coefs and trace_batch, with trace_batch
    /// indexed by pois
    /// proj_coefs and pois must have the same length.
    #[inline(never)]
    fn project_inner_loop(
        &self,
        acc: &mut [f64; N],
        trace_batch: &[AA<N>],
        proj_coefs: &[f64],
        pois: &[u16],
    ) {
        for (coef, poi) in proj_coefs.iter().zip(pois.iter()) {
            let samples = &trace_batch[*poi as usize];
            for i in 0..N {
                acc[i] += *coef * (samples.0[i] as f64);
            }
        }
    }

    // Project the traces for all the vars
    // traces: input traces of shape (ntraces, ns)
    // returns: a vec of size nv, where every element is an array of shape (ntraces, p) contiaining
    // the projected traces for a variable.
    pub fn project(&self, traces: ArrayView2<i16>) -> Vec<Array2<f64>> {
        let n = traces.shape()[0];
        let mut p_traces = vec![Array2::<f64>::zeros((n, self.p)); self.n_vars() as usize];
        let traces_batched = self.poi_map.select_batches::<N>(traces);
        for var_block in (0..(self.poi_blocks.len())).range_chunks(self.var_block_size()) {
            // nbatches arrays of shape (len(var_block), p)), where a[i,j] contains the
            // batch_traces_projected[v_i, j], for the variable var_block[v_i] at the j-th
            // dimension in the linear subspace.
            let v_p_traces = self.project_traces(var_block.clone(), &traces_batched);
            for (bi, p_batch_trace) in v_p_traces.into_iter().enumerate() {
                for (vi, v) in var_block.clone().into_iter().enumerate() {
                    for pi in 0..self.p {
                        let batch = p_batch_trace[[vi, pi]];
                        for ni in (0..(cmp::min(n - (bi * N), batch.len()))) {
                            p_traces[v][[(bi * N) + ni, pi]] = batch[ni];
                        }
                    }
                }
            }
        }
        p_traces
    }

    /// For a var, write in res the log likelihoods given the projected traces
    /// p_traces: projected traces, length: p
    /// res: (N, nc)
    fn compute_ll_thread_loop(
        &self,
        var: usize,
        mut res: ArrayViewMut2<f64>,
        p_traces: &[[f64; N]],
    ) {
        for (mut res, pk, omega) in izip!(
            res.axis_iter_mut(Axis(1)),
            self.lda_states[var].pk.iter(),
            self.lda_states[var].omega.outer_iter()
        ) {
            let log_likelihood = self.ll_from_p_traces(p_traces, *pk, omega);
            for i in 0..res.len() {
                res[i] = log_likelihood[i];
            }
        }
    }

    /// compute omega*p_trace + pk for each trace in p_traces
    /// p_traces len: p
    #[inline(always)]
    fn ll_from_p_traces(&self, p_traces: &[[f64; N]], pk: f64, omega: ArrayView1<f64>) -> [f64; N] {
        let omega = omega.as_slice().unwrap();
        let mut log_likelihood = [pk; N];
        for p_i in 0..self.p {
            for i in 0..N {
                log_likelihood[i] += p_traces[p_i][i] * omega[p_i];
            }
        }
        log_likelihood
    }
    fn n_vars(&self) -> Var {
        self.lda_states.len() as Var
    }
    /// POI blocks for computing projections.
    fn poi_block_ranges(&self) -> impl Iterator<Item = Range<usize>> {
        (0..(self.poi_map.len() as usize)).range_chunks(POI_BLOCK_SIZE)
    }

    /// Make a new multi LDA with a subset of the variables
    pub fn select_vars(&self, vars: &[Var]) -> Result<Self> {
        if vars.iter().any(|v| *v >= self.n_vars()) {
            return Err(ScalibError::VarOutOfBound);
        }
        let new_map = self.poi_map.select_vars(vars)?;
        Ok(Self {
            nc: self.nc,
            p: self.p,
            poi_blocks: new_map.poi_blocks(),
            poi_map: Arc::new(new_map),
            // Since POIs are kept sorted, no need to modify kept LDAs.
            lda_states: vars
                .iter()
                .map(|v| self.lda_states[*v as usize].clone())
                .collect(),
        })
    }
}
