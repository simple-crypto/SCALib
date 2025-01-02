mod batched_traces;
mod cov_pairs;
mod poi_map;
mod sparse_trace_sums;

pub type Class = u16;
pub type Var = u16;

use std::ops::Mul;

use cov_pairs::{CovAcc, CovPairs};
use poi_map::PoiMap;
use sparse_trace_sums::{SparseTraceSums, SparseTraceSumsState};

use itertools::{izip, Itertools};
use ndarray::{azip, Array2, ArrayView2, Axis};

type Result<T> = std::result::Result<T, ()>;

#[derive(Debug, Clone)]
struct MultiLdaConf {
    ns: u32,
    nv: Var,
    // Pois for each var
    pois: Vec<Vec<u32>>,
    pois_mapped: Vec<Vec<u32>>,
    cov_pois_offsets: Vec<usize>,
    poi_map: PoiMap,
    trace_sums: SparseTraceSums,
    cov_pois: CovPairs,
}

impl MultiLdaConf {
    fn new(ns: u32, nc: Class, mut pois: Vec<Vec<u32>>) -> Result<Self> {
        // Sort POIs: required for SparseTraceSums and has not impact on the LDA result.
        for pois in pois.iter_mut() {
            pois.sort_unstable();
        }
        let nv: Var = pois.len().try_into().map_err(|_| ())?;
        let poi_map = PoiMap::new(ns, pois.iter().flat_map(|x| x.iter().copied()))?;
        let pois_mapped = pois
            .iter()
            .map(|x| x.iter().map(|y| poi_map.to_new(*y).unwrap()).collect_vec())
            .collect_vec();
        let trace_sums = SparseTraceSums::new(ns, nv, nc, pois.as_slice());
        let mapped_pairs = pois_mapped
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
            ns,
            nv,
            pois,
            pois_mapped,
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
        Self::pairs_n(self.pois[var as usize].len() as u32)
    }
    fn npairs_n(n: usize) -> usize {
        n * (n + 1) / 2
    }
    fn npairs_var(&self, var: Var) -> usize {
        Self::npairs_n(self.pois_mapped[var as usize].len())
    }
    fn var_covpoi_pairs_idxs(&self, var: Var) -> std::ops::Range<usize> {
        let start = self.cov_pois_offsets[var as usize];
        start..(start + self.npairs_var(var))
    }
    fn npois_var(&self, var: Var) -> usize {
        self.pois_mapped[var as usize].len()
    }
}

#[derive(Debug, Clone)]
struct MultiLdaState {
    n_traces: u32,
    trace_sums: SparseTraceSumsState,
    cov_acc: CovAcc,
}

#[derive(Debug, Clone)]
pub struct LdaMatrices {
    pub s_b: Array2<f64>,
    pub s_w: Array2<f64>,
    pub mus: Array2<f64>,
}
impl LdaMatrices {
    fn from_seqs(npois: usize, s_b: Vec<f64>, s_w: Vec<f64>, mus: Array2<f64>) -> Self {
        Self {
            s_b: Self::seq2mat(npois, s_b.as_slice()),
            s_w: Self::seq2mat(npois, s_w.as_slice()),
            mus,
        }
    }
    fn seq2mat(n: usize, seq: &[f64]) -> Array2<f64> {
        let mut res = Array2::zeros((n, n));
        for ((i, j), x) in MultiLdaConf::pairs_n(n as u32).zip(seq.iter()) {
            res[(i as usize, j as usize)] = *x;
            res[(j as usize, i as usize)] = *x;
        }
        res
    }
    fn to_tuple(self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        (self.s_w, self.s_b, self.mus)
    }
}

impl MultiLdaState {
    fn new(multi_lda: &MultiLdaConf) -> Self {
        Self {
            n_traces: 0,
            trace_sums: SparseTraceSumsState::new(&multi_lda.trace_sums),
            cov_acc: CovAcc::new(&multi_lda.cov_pois),
        }
    }
    fn update(
        &mut self,
        multi_lda: &MultiLdaConf,
        traces: ArrayView2<i16>,
        y: ArrayView2<Class>,
    ) -> Result<()> {
        self.n_traces = self
            .n_traces
            .checked_add(traces.shape()[0].try_into().map_err(|_| ())?)
            .ok_or(())?;
        self.trace_sums.update(&multi_lda.trace_sums, traces, y);
        self.cov_acc
            .update_para(&multi_lda.poi_map, &multi_lda.cov_pois, traces)?;
        Ok(())
    }
    fn s_b_u(&self, multi_lda: &MultiLdaConf, var: Var) -> (Vec<i64>, Vec<f64>) {
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
    fn compute_matrices_var(&self, multi_lda: &MultiLdaConf, var: Var) -> Result<LdaMatrices> {
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
            return Err(());
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
        Ok(LdaMatrices::from_seqs(
            multi_lda.npois_var(var),
            s_w,
            s_b,
            mus,
        ))
    }

    fn compute_matrices(&self, multi_lda: &MultiLdaConf) -> Result<Vec<LdaMatrices>> {
        (0..multi_lda.nv)
            .map(|var| self.compute_matrices_var(multi_lda, var))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MultiLda {
    conf: MultiLdaConf,
    state: MultiLdaState,
}

impl MultiLda {
    pub fn new(ns: u32, nc: Class, pois: Vec<Vec<u32>>) -> Result<Self> {
        let conf = MultiLdaConf::new(ns, nc, pois)?;
        let state = MultiLdaState::new(&conf);
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
    pub fn pois(&self) -> &Vec<Vec<u32>> {
        &self.conf.pois
    }
}
