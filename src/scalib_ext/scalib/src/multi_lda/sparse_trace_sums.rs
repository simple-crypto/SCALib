use ndarray::{s, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use super::{Class, Var};

// TODO: accumulator on 32-bit (tmp).
// TODO we can get better performance by batching the "ns" axis (possibly with SIMD gather).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTraceSums {
    nv: Var,
    nc: Class,
    // Number of pois for each var
    n_pois: Vec<usize>,
    // For each point in trace, the list of vars for which we have that POI.
    vars_per_poi: Vec<Vec<Var>>,
}
impl SparseTraceSums {
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
        Self {
            nv,
            nc,
            n_pois,
            vars_per_poi,
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
    pub fn new(acc: &SparseTraceSums) -> Self {
        let n_traces = Array2::zeros((acc.nv as usize, acc.nc as usize));
        let sums = (0..acc.nv)
            .map(|i| Array2::zeros((acc.n_pois[i as usize], acc.nc as usize)))
            .collect();
        Self { n_traces, sums }
    }
    pub fn update(&mut self, acc: &SparseTraceSums, traces: ArrayView2<i16>, y: ArrayView2<Class>) {
        // Update n_traces
        for (mut n_traces, y) in self.n_traces.outer_iter_mut().zip(y.axis_iter(Axis(1))) {
            for y in y.iter() {
                n_traces[*y as usize] += 1;
            }
        }
        // Update sums
        let mut var_poi_count = vec![0u32; acc.nv as usize];
        for (samples, vars) in traces.axis_iter(Axis(1)).zip(acc.vars_per_poi.iter()) {
            for var in vars {
                let sums = &mut self.sums[*var as usize];
                let y = y.slice(s![.., *var as usize]);
                let poi_id = var_poi_count[*var as usize] as usize;
                var_poi_count[*var as usize] += 1;
                for (sample, y) in samples.iter().zip(y.iter()) {
                    sums[(poi_id, *y as usize)] += *sample as i64;
                }
            }
        }
    }
}
