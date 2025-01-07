mod batched_traces;
mod cov_pairs;
mod poi_map;
mod sparse_trace_sums;

use std::ops::Range;
use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::lda::{softmax, LDA};
use crate::{Result, ScalibError};
use batched_traces::BatchedTraces;
use cov_pairs::{CovAcc, CovPairs};
use poi_map::PoiMap;
use sparse_trace_sums::{SparseTraceSums, SparseTraceSumsState};

pub type Class = u16;
pub type Var = u16;

use itertools::{izip, Itertools};
use ndarray::{azip, Array2, Array3, ArrayView2, ArrayViewMut2, Axis};

trait RangeExt {
    type Idx;
    fn range_chunks(self, size: Self::Idx) -> impl Iterator<Item = Self>;
}

impl RangeExt for Range<usize> {
    type Idx = usize;
    fn range_chunks(self, size: Self::Idx) -> impl Iterator<Item = Self> {
        let l = self.len();
        (0..(l.div_ceil(size))).map(move |i| (i * size..std::cmp::min((i + 1) * size, l)))
    }
}

trait ArrayBaseExt<A, D> {
    fn clone_row_major(&self) -> ndarray::Array<A, D>;
}
impl<A, S, D> ArrayBaseExt<A, D> for ndarray::ArrayBase<S, D>
where
    A: Clone,
    S: ndarray::RawData<Elem = A> + ndarray::Data,
    D: ndarray::Dimension,
{
    fn clone_row_major(&self) -> ndarray::Array<A, D> {
        let res = ndarray::Array::<A, D>::build_uninit(self.dim(), |a| {
            self.assign_to(a);
        });
        unsafe { res.assume_init() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultiLdaAccConf {
    nv: Var,
    nc: Class,
    // Pois for each var
    cov_pois_offsets: Vec<usize>,
    poi_map: Arc<PoiMap>,
    trace_sums: SparseTraceSums,
    cov_pois: CovPairs,
}

impl MultiLdaAccConf {
    fn new(ns: u32, nc: Class, mut pois: Vec<Vec<u32>>) -> Result<Self> {
        // Sort POIs: required for SparseTraceSums and has not impact on the LDA result.
        for pois in pois.iter_mut() {
            pois.sort_unstable();
        }
        let nv: Var = pois
            .len()
            .try_into()
            .map_err(|_| ScalibError::TooManyPois)?;
        let poi_map = Arc::new(PoiMap::new(ns, &pois)?);
        let trace_sums = SparseTraceSums::new(ns, nv, nc, pois.as_slice());
        let mapped_pairs = poi_map
            .new_pois_vars()
            .iter()
            .flat_map(|pois| {
                Self::pairs_n(pois.len() as u32).map(|(i, j)| (pois[i as usize], pois[j as usize]))
            })
            .collect_vec();
        let cov_pois_offsets = pois
            .iter()
            .scan(0, |acc, x| {
                let res = *acc;
                *acc += Self::npairs_n(x.len());
                Some(res)
            })
            .collect_vec();

        let cov_pois = CovPairs::new(poi_map.len(), mapped_pairs.as_slice())?;
        Ok(Self {
            nv,
            nc,
            cov_pois_offsets,
            poi_map,
            trace_sums,
            cov_pois,
        })
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
    fn npairs_var(&self, var: Var) -> usize {
        Self::npairs_n(self.poi_map.n_pois(var))
    }
    fn var_covpoi_pairs_idxs(&self, var: Var) -> Range<usize> {
        let start = self.cov_pois_offsets[var as usize];
        start..(start + self.npairs_var(var))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultiLdaAccState {
    n_traces: u32,
    trace_sums: SparseTraceSumsState,
    cov_acc: CovAcc,
}

impl MultiLdaAccState {
    fn new(multi_lda: &MultiLdaAccConf) -> Self {
        Self {
            n_traces: 0,
            trace_sums: SparseTraceSumsState::new(&multi_lda.trace_sums),
            cov_acc: CovAcc::new(&multi_lda.cov_pois),
        }
    }
    fn update(
        &mut self,
        multi_lda: &MultiLdaAccConf,
        traces: ArrayView2<i16>,
        y: ArrayView2<Class>,
    ) -> Result<()> {
        self.n_traces = self
            .n_traces
            .checked_add(
                traces.shape()[0]
                    .try_into()
                    .map_err(|_| ScalibError::TooManyPois)?,
            )
            .ok_or(ScalibError::TooManyTraces)?;
        self.trace_sums.update(&multi_lda.trace_sums, traces, y);
        self.cov_acc
            .update(&multi_lda.poi_map, &multi_lda.cov_pois, traces)?;
        Ok(())
    }
    fn s_b_u(&self, multi_lda: &MultiLdaAccConf, var: Var) -> (Vec<i64>, Vec<f64>) {
        let sums = &self.trace_sums.sums[var as usize];
        let n_traces = self.trace_sums.n_traces.index_axis(Axis(0), var as usize);
        let mut s_b_u_int = vec![0; multi_lda.npairs_var(var)];
        let mut s_b_u_frac = vec![0.0; multi_lda.npairs_var(var)];
        for (class_sum, n_traces) in sums.axis_iter(Axis(1)).zip(n_traces.iter()) {
            let class_sum = class_sum.iter().copied().collect_vec();
            let n_traces_d = quickdiv::DivisorI128::new(*n_traces as i128);
            for (k, (i, j)) in multi_lda.var_pairs(var).enumerate() {
                let x = (class_sum[i as usize] as i128) * (class_sum[j as usize] as i128);
                // No overflow: this is bounded by (i16::MIN)**2*n_traces
                let x_int = (x / n_traces_d) as i64;
                // No overflow: this is bounded by (i16::MIN)**2*n_traces
                let x_rem = (x % n_traces_d) as i32;
                // No overflow: this is bounded by (i16::MIN)**2*n_tot_traces
                s_b_u_int[k] += x_int;
                s_b_u_frac[k] += (x_rem as f64) / (*n_traces as f64);
            }
        }
        (s_b_u_int, s_b_u_frac)
    }
    fn compute_matrices_var(&self, multi_lda: &MultiLdaAccConf, var: Var) -> Result<LdaMatrices> {
        // LDA matrices computation.
        // x == data, xi == data with class i
        // n == number of traces, ni == number of traces with class i
        // Sx(.) == sum over all traces
        // Si(.) == sum over classes
        // Sxi(.) == sum over traces of class i
        // mu = Sx(x)/n (average)
        // mui = Sxi(xi)/ni (class average)
        // cmui = mui-mu (centered class average)
        //
        // # Classic LDA definitions:
        // - between-class scatter: s_b = Si(ni*cmui**2)
        // - within-class scatter: s_w = Si(Sxi((xi-mui)**2))
        // - total scatter: s_t = Sx((x-mu)**2)
        //
        // # Simplified expressions:
        // let
        // - uncentered total scatter s_t_u = Sx(x**2),
        // - uncentered between-class scatter s_b_u = Si(Sxi(xi)**2/ni),
        // - total of all traces ts = Sx(x)
        // we have
        // s_b = s_b_u - ts**2/n
        // s_w = s_t_u - s_b_u
        // where s_t_u is computed by CovAcc and s_b_u can be derived from per-class sums.
        let sums = &self.trace_sums.sums[var as usize];
        let n_traces = self.trace_sums.n_traces.index_axis(Axis(0), var as usize);
        if n_traces.iter().any(|n| *n == 0) {
            return Err(ScalibError::EmptyClass);
        }
        // Total scatter offset by mu**2*n_tot.
        let s_t_u = multi_lda
            .var_covpoi_pairs_idxs(var)
            .map(|k| self.cov_acc.scatter[multi_lda.cov_pois.pair_to_new_idx[k] as usize])
            .collect_vec();
        let (s_b_u_int, s_b_u_frac) = self.s_b_u(multi_lda, var);
        // No overflow: intermediate bounded by 2*(i16::MIN)**2*n_tot_traces
        let s_w = izip!(s_t_u.into_iter(), s_b_u_int.iter(), s_b_u_frac.iter())
            .map(|(t, b_i, b_f)| ((t - *b_i) as f64) - *b_f)
            .collect_vec();
        // Offset for s_b.
        let ts = sums.sum_axis(Axis(1));
        let tot_ntraces_d = quickdiv::DivisorI128::new(self.n_traces as i128);
        let s_b = izip!(
            s_b_u_int.into_iter(),
            s_b_u_frac.into_iter(),
            multi_lda.var_pairs(var)
        )
        .map(|(b_i, b_f, (i, j))| {
            let x = (ts[i as usize] as i128) * (ts[j as usize] as i128);
            let c_i = (x / tot_ntraces_d) as i64;
            let c_f = (((x % tot_ntraces_d) as i32) as f64) / (self.n_traces as f64);
            ((b_i - c_i) as f64) + (b_f - c_f)
        })
        .collect_vec();
        let mus = azip!(sums.t())
            .and_broadcast(n_traces.insert_axis(Axis(1)))
            .map_collect(|s, n| (*s as f64) / (*n as f64));
        Ok(LdaMatrices {
            s_w: LdaMatrices::seq2mat(multi_lda.poi_map.n_pois(var), s_w.as_slice()),
            s_b: LdaMatrices::seq2mat(multi_lda.poi_map.n_pois(var), s_b.as_slice()),
            mus,
            n_traces: self.n_traces,
        })
    }

    fn compute_matrices(&self, multi_lda: &MultiLdaAccConf) -> Result<Vec<LdaMatrices>> {
        (0..multi_lda.nv)
            .map(|var| self.compute_matrices_var(multi_lda, var))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLdaAcc {
    conf: MultiLdaAccConf,
    state: MultiLdaAccState,
}

impl MultiLdaAcc {
    pub fn new(ns: u32, nc: Class, pois: Vec<Vec<u32>>) -> Result<Self> {
        let conf = MultiLdaAccConf::new(ns, nc, pois)?;
        let state = MultiLdaAccState::new(&conf);
        Ok(Self { conf, state })
    }
    pub fn update(&mut self, traces: ArrayView2<i16>, y: ArrayView2<Class>) -> Result<()> {
        self.state.update(&self.conf, traces, y)
    }
    pub fn ntraces(&self) -> u32 {
        self.state.n_traces
    }
    pub fn get_matrices(&self) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>> {
        Ok(self
            .state
            .compute_matrices(&self.conf)?
            .into_iter()
            .map(LdaMatrices::to_tuple)
            .collect())
    }
    pub fn lda(&self, p: u32) -> Result<MultiLda> {
        MultiLda::new(&self.conf, &self.state.compute_matrices(&self.conf)?, p)
    }
}

#[derive(Debug, Clone)]
pub struct LdaMatrices {
    pub s_b: Array2<f64>,
    pub s_w: Array2<f64>,
    pub mus: Array2<f64>,
    pub n_traces: u32,
}

impl LdaMatrices {
    fn seq2mat(n: usize, seq: &[f64]) -> Array2<f64> {
        let mut res = Array2::zeros((n, n));
        for ((i, j), x) in MultiLdaAccConf::pairs_n(n as u32).zip(seq.iter()) {
            res[(i as usize, j as usize)] = *x;
            res[(j as usize, i as usize)] = *x;
        }
        res
    }
    fn to_tuple(self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        (self.s_w, self.s_b, self.mus)
    }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLda {
    nc: Class,
    p: usize,
    ldas: Vec<LDA>,
    poi_map: Arc<PoiMap>,
    // indexing: [var][poi_block][poi_offset_in_block]
    poi_blocks: Vec<Vec<Vec<u16>>>,
}

impl MultiLda {
    fn new(conf: &MultiLdaAccConf, matrices: &[LdaMatrices], p: u32) -> Result<Self> {
        let ldas = matrices.iter().map(|m| m.lda(p)).collect::<Result<_>>()?;
        let poi_blocks = conf.poi_map.poi_blocks();
        let p = p as usize;
        Ok(Self {
            nc: conf.nc,
            p,
            ldas,
            poi_map: conf.poi_map.clone(),
            poi_blocks,
        })
    }
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
    /// return the probability of each of the possible value for leakage samples
    /// traces with shape (n,ns)
    /// return prs with shape (nv,n,nc). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, traces: ArrayView2<i16>) -> Array3<f64> {
        let mut res = Array3::zeros((self.poi_blocks.len(), traces.shape()[0], self.nc as usize));

        let traces_batched = BatchedTraces::<N>::new(&self.poi_map, traces).get_batches();

        for var_block in (0..(self.poi_blocks.len())).range_chunks(self.var_block_size()) {
            let mut tmp = vec![
                Array2::from_elem((var_block.len(), self.p), [0.0f64; N]);
                traces_batched.len()
            ];
            let projections = var_block
                .clone()
                .map(|var| self.ldas[var].projection.t().clone_row_major())
                .collect_vec();
            (traces_batched.outer_iter(), tmp.as_mut_slice())
                .into_par_iter()
                .for_each(|(trace_batch, tmp)| {
                    self.project_thread_loop(
                        var_block.clone(),
                        &projections,
                        tmp,
                        trace_batch.as_slice().unwrap(),
                    );
                });
            for (var_i, var) in var_block.clone().enumerate() {
                let mut res = res.index_axis_mut(Axis(0), var);
                let lda = &self.ldas[var];
                let omega = lda.omega.t().clone_row_major();
                (res.axis_chunks_iter_mut(Axis(0), N), tmp.as_slice())
                    .into_par_iter()
                    .for_each(|(res, tmp)| {
                        self.compute_proba_thread_loop(var_i, lda, &omega, res, tmp);
                    });
            }
        }
        res
    }

    fn project_thread_loop(
        &self,
        var_block: Range<usize>,
        projections: &[Array2<f64>],
        tmp: &mut Array2<[f64; N]>,
        trace_batch: &[batched_traces::AA<N>],
    ) {
        tmp.fill([0.0; N]);
        for (poi_block, poi_block_range) in self.poi_block_ranges().enumerate() {
            let trace_batch = &trace_batch[poi_block_range];
            for (var, mut tmp) in var_block.clone().zip(tmp.outer_iter_mut()) {
                let pois = self.poi_blocks[var][poi_block].as_slice();
                let proj = &projections[var];
                for (tmp, coefs) in tmp.iter_mut().zip(proj.outer_iter()) {
                    let coefs = coefs.as_slice().unwrap();
                    self.project_inner_loop(tmp, trace_batch, coefs, pois);
                }
            }
        }
    }
    #[inline(never)]
    fn project_inner_loop(
        &self,
        acc: &mut [f64; N],
        trace_batch: &[batched_traces::AA<N>],
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

    fn compute_proba_thread_loop(
        &self,
        var_i: usize,
        lda: &LDA,
        omega: &Array2<f64>,
        mut res: ArrayViewMut2<f64>,
        tmp: &Array2<[f64; N]>,
    ) {
        let tmp = tmp.index_axis(Axis(0), var_i);
        let tmp = tmp.as_slice().unwrap();
        for (mut res, pk, omega) in izip!(
            res.axis_iter_mut(Axis(1)),
            lda.pk.iter(),
            omega.outer_iter()
        ) {
            let omega = omega.as_slice().unwrap();
            let mut scores = [*pk; N];
            for p_i in 0..self.p {
                for i in 0..N {
                    scores[i] += tmp[p_i][i] * omega[p_i];
                }
            }
            for i in 0..res.len() {
                res[i] = scores[i];
            }
        }
        for res in res.outer_iter_mut() {
            softmax(res);
        }
    }
    fn poi_block_ranges(&self) -> impl Iterator<Item = Range<usize>> {
        (0..(self.poi_map.len() as usize)).range_chunks(POI_BLOCK_SIZE)
    }
}
