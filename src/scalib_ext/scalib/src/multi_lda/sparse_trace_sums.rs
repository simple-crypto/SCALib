use itertools::{izip, Itertools};
use ndarray::{Array2, ArrayView2, Axis};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use super::{ArrayBaseExt, Class, RangeExt, Var};

// TODO: accumulator on 32-bit (tmp).
// TODO we can get better performance by batching the "ns" axis (possibly with SIMD gather).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTraceSumsConf {
    nv: Var,
    nc: Class,
    // Number of pois for each var
    n_pois: Vec<usize>,
    // For each point in trace, the list of vars for which we have that POI.
    vars_per_poi: Vec<Vec<Var>>,
    // Id of POI for each var in vars_per_poi
    poi_var_id: Vec<Vec<u16>>,
    poi_v_poiid_chunks: Vec<(Vec<u32>, Vec<Var>, Vec<usize>)>,
    traces_block_size: usize,
}
impl SparseTraceSumsConf {
    /// pois: for each var, list of pois
    pub fn new(ns: u32, nv: u16, nc: u16, pois: &[Vec<u32>]) -> Self {
        for pois in pois {
            assert!(pois.is_sorted(), "POIs not sorted");
        }
        let n_pois = pois.iter().map(Vec::len).collect();
        let mut vars_per_poi = vec![vec![]; ns as usize];
        for (var, pois_v) in pois.iter().enumerate() {
            for poi in pois_v {
                vars_per_poi[*poi as usize].push(var as Var);
            }
        }
        let mut var_poi_count = vec![0u16; nv as usize];
        let poi_var_id = vars_per_poi
            .iter()
            .map(|vars| {
                vars.iter()
                    .map(|var| {
                        let res = var_poi_count[*var as usize];
                        var_poi_count[*var as usize] = res.checked_add(1).unwrap();
                        res
                    })
                    .collect()
            })
            .collect();
        // Available high-bandwidth L3 cache size (ideally to be tuned per-CPU).
        // (conservative for modern machines: we do not want to fill it too much).
        const L3_CACHE_SIZE: usize = 1 << 23;
        // Assume we can perform 1 sum operation per clock cycle thanks for super-scalar.
        // Assume we can load 1 byte every 4 clock cycles, since 1 sum accumulator is 8 bytes,
        // we can load 1 accumulator every 32 operations.
        // Therefore, block size along traces should be at least 32*nc.
        const TRACES_BLOCK_SIZE_PER_CLASS: usize = 32;
        let traces_block_size = TRACES_BLOCK_SIZE_PER_CLASS * usize::from(nc);
        // We evenly split the L3 cache between class storage and trace storage.
        // At 2 bytes per element, we get:
        let var_block_size = L3_CACHE_SIZE / 2 / 2 / traces_block_size;
        let poi_block_size = L3_CACHE_SIZE / 2 / 2 / traces_block_size;
        let poi_v_poiid_chunks = (0..usize::from(nv))
            .range_chunks(var_block_size)
            .flat_map(|var_block| {
                let poisv = itertools::sorted_unstable(var_block.flat_map(|v| {
                    pois[v]
                        .iter()
                        .enumerate()
                        .map(move |(poi_id, p)| (*p, v as u16, poi_id))
                }));
                let poisv_per_poi = poisv.into_iter().chunk_by(|(p, _, _)| *p);
                let poisv_per_poi_chunks = poisv_per_poi.into_iter().chunks(poi_block_size);
                poisv_per_poi_chunks
                    .into_iter()
                    .map(move |poisv_per_poi_chunk| {
                        poisv_per_poi_chunk
                            .flat_map(|(_, poisv)| poisv)
                            .multiunzip()
                    })
                    .collect_vec()
            })
            .collect();

        Self {
            nv,
            nc,
            n_pois,
            vars_per_poi,
            poi_var_id,
            poi_v_poiid_chunks,
            traces_block_size,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTraceSumsState {
    /// Sum of all the traces corresponding to each class. Shape: (nv) (npois[i], nc).
    pub sums: Vec<Array2<i64>>,
    /// Number of traces for each class. Shape: (nv, nc).
    pub n_traces: Array2<u32>,
}

impl SparseTraceSumsState {
    pub fn new(conf: &SparseTraceSumsConf) -> Self {
        let n_traces = Array2::zeros((conf.nv as usize, conf.nc as usize));
        let sums = (0..conf.nv)
            .map(|i| Array2::zeros((conf.n_pois[i as usize], conf.nc as usize)))
            .collect();
        Self { n_traces, sums }
    }
    pub fn update(
        &mut self,
        conf: &SparseTraceSumsConf,
        traces: ArrayView2<i16>,
        y: ArrayView2<Class>,
    ) {
        assert_eq!(y.shape()[1], conf.nv as usize);
        let mut sums = Self::split_sums(&mut self.sums, conf);
        for (traces, y) in izip!(
            traces.axis_chunks_iter(Axis(0), conf.traces_block_size),
            y.axis_chunks_iter(Axis(0), conf.traces_block_size),
        ) {
            let traces = traces.t().clone_row_major();
            assert_eq!(y.shape()[1], conf.nv as usize);
            let y = y.t().clone_row_major();
            assert_eq!(y.shape()[0], conf.nv as usize);
            for (mut n_traces, y) in self.n_traces.outer_iter_mut().zip(y.outer_iter()) {
                for y in y.iter() {
                    n_traces[*y as usize] += 1;
                }
            }
            Self::update_trace_chunk(conf, &mut sums, &traces, &y);
        }
    }
    fn split_sums<'s>(
        sums: &'s mut Vec<Array2<i64>>,
        conf: &SparseTraceSumsConf,
    ) -> Vec<Vec<&'s mut [i64]>> {
        let mut sums_per_poi_sep_list = sums
            .iter_mut()
            .map(|sums| sums.axis_iter_mut(Axis(0)).map(Some).collect_vec())
            .collect_vec();
        conf.poi_v_poiid_chunks
            .iter()
            .map(|(_, vars, poi_ids)| {
                izip!(vars.iter(), poi_ids.iter())
                    .map(|(var, poi_id)| {
                        sums_per_poi_sep_list[*var as usize][*poi_id as usize]
                            .take()
                            .unwrap()
                            .into_slice()
                            .unwrap()
                    })
                    .collect_vec()
            })
            .collect_vec()
    }
    fn update_trace_chunk(
        conf: &SparseTraceSumsConf,
        sums: &mut [Vec<&mut [i64]>],
        traces: &Array2<i16>,
        y: &Array2<Class>,
    ) {
        for ((pois, vars, _), sums) in izip!(conf.poi_v_poiid_chunks.iter(), sums.iter_mut()) {
            Self::update_inner(sums, vars, pois, traces, y);
        }
    }
    fn update_inner(
        sums: &mut [&mut [i64]],
        vs: &[Var],
        pois: &[u32],
        traces: &Array2<i16>,
        y: &Array2<Class>,
    ) {
        (vs, pois, sums).into_par_iter().for_each(|(v, poi, sums)| {
            let samples = traces.index_axis(Axis(0), *poi as usize);
            let samples = samples.as_slice().unwrap();
            assert!((*v as usize) < y.shape()[0]);
            let y = y.index_axis(Axis(0), usize::from(*v));
            let y = y.as_slice().unwrap();
            for (s, y) in samples.iter().zip(y.iter()) {
                sums[usize::from(*y)] += i64::from(*s);
            }
        });
    }
}
