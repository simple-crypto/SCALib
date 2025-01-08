use itertools::{izip, Itertools};
use ndarray::{Array2, ArrayView2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{Result, ScalibError};

use super::batched_traces::{BatchedTraces, AA};
use super::poi_map::PoiMap;

const N: usize = 64;
const MAX_CHUNK_SIZE: usize = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovPairs {
    /// Pairs are pairs of POIs for which the covariance is computed.
    /// Values in `sorted_pairs` are sorted such that consecutive pairs have good memory locality
    /// (i.e., tend to have the same POIs).
    sorted_pairs: Vec<(u32, u32)>,
    /// Ranges in `sorted_pairs` corresponding to values that should be computed together for
    /// good cache locality.
    chunks: Vec<std::ops::Range<usize>>,
    /// Mapping between orignal pair indices and the corresponding pair in `sorted_pairs`.
    pub pair_to_new_idx: Vec<u32>,
}

impl CovPairs {
    pub fn new(ns: usize, pairs: &[(u32, u32)]) -> Result<Self> {
        let (sorted_pairs, pair_to_new_idx, chunks) = if ns <= MAX_CHUNK_SIZE {
            #[allow(clippy::single_range_in_vec_init)]
            (
                pairs.to_vec(),
                (0..(pairs
                    .len()
                    .try_into()
                    .map_err(|_| ScalibError::TooManyPois)?))
                    .collect(),
                vec![0..pairs.len()],
            )
        } else {
            let (pair_to_old_idx, chunks) = chunk_pairs(pairs, ns, MAX_CHUNK_SIZE)?;
            let mut pair_to_new_idx = vec![0; pairs.len()];
            for (i, k) in pair_to_old_idx.iter().enumerate() {
                pair_to_new_idx[*k as usize] = i as u32;
            }
            let sorted_pairs = pair_to_new_idx.iter().map(|k| pairs[*k as usize]).collect();
            (sorted_pairs, pair_to_new_idx, chunks)
        };
        Ok(Self {
            sorted_pairs,
            chunks,
            pair_to_new_idx,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovAcc {
    pub tot_n_traces: u32,
    pub scatter: Vec<i64>,
}

impl CovAcc {
    pub fn new(cov_pois: &CovPairs) -> Self {
        Self {
            tot_n_traces: 0,
            scatter: vec![0; cov_pois.sorted_pairs.len()],
        }
    }
    pub fn update(
        &mut self,
        poi_map: &PoiMap,
        pois: &CovPairs,
        traces: ArrayView2<i16>,
    ) -> Result<()> {
        assert!(traces.is_standard_layout());
        let n_traces: u32 = traces.shape()[0]
            .try_into()
            .map_err(|_| ScalibError::TooManyTraces)?;
        self.tot_n_traces = self
            .tot_n_traces
            .checked_add(n_traces)
            .ok_or(ScalibError::TooManyTraces)?;
        let traces = BatchedTraces::<N>::new(poi_map, traces).get_batches();
        let split_at = pois.chunks[1..].iter().map(|r| r.start);
        let scatter_chunks = multi_split_at_mut(self.scatter.as_mut_slice(), split_at.clone());
        let pair_chunks = multi_split_at(pois.sorted_pairs.as_slice(), split_at);
        (scatter_chunks, pair_chunks)
            .into_par_iter()
            .for_each(|(scatter_chunk, pair_chunk)| {
                Self::update_chunk(&traces, scatter_chunk, pair_chunk);
            });
        Ok(())
    }
    fn update_chunk(traces: &Array2<AA<N>>, scatter_chunk: &mut [i64], pair_chunk: &[(u32, u32)]) {
        assert!(traces.is_standard_layout());
        for batch in traces.outer_iter() {
            for (scatter, (i, j)) in izip!(scatter_chunk.iter_mut(), pair_chunk.iter()) {
                *scatter += sum_prod(&batch[*i as usize], &batch[*j as usize]);
            }
        }
    }
}

fn sum_prod<'a>(x: &'a AA<N>, y: &'a AA<N>) -> i64 {
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        const {
            assert!(N % 16 == 0);
        };
        use std::arch::x86_64::{
            __m256i, _mm256_add_epi64, _mm256_cvtepi32_epi64, _mm256_extracti128_si256,
            _mm256_madd_epi16, _mm256_setzero_si256, _mm_add_epi64, _mm_cvtsi128_si64x,
            _mm_shuffle_epi32,
        };
        unsafe {
            // Transmute safety: AA is 32-byte aligned, we keep the same lifetime.
            let x: &'a [__m256i; N / 16] = core::mem::transmute(x);
            let y: &'a [__m256i; N / 16] = core::mem::transmute(y);
            let mut res: __m256i = _mm256_setzero_si256();
            for k in 0..(N / 16) {
                let tmp = _mm256_madd_epi16(x[k], y[k]);
                let tmp0 = _mm256_extracti128_si256(tmp, 0);
                let tmp1 = _mm256_extracti128_si256(tmp, 1);
                let tmp0 = _mm256_cvtepi32_epi64(tmp0);
                let tmp1 = _mm256_cvtepi32_epi64(tmp1);
                let tmp_sum = _mm256_add_epi64(tmp0, tmp1);
                res = _mm256_add_epi64(res, tmp_sum);
            }
            let res0 = _mm256_extracti128_si256(res, 0);
            let res1 = _mm256_extracti128_si256(res, 1);
            let res = _mm_add_epi64(res0, res1);
            let res_t = _mm_shuffle_epi32(res, 0xee);
            let res = _mm_add_epi64(res, res_t);
            _mm_cvtsi128_si64x(res)
        }
    } else {
        // Default, non-intrinsics
        let mut res = 0;
        for k in 0..N {
            res += ((x.0[k] as i32) * (y.0[k] as i32)) as i64;
        }
        res
    }
}

fn chunk_pairs(
    pairs: &[(u32, u32)],
    ns: usize,
    max_chunk_size: usize,
) -> Result<(Vec<u32>, Vec<std::ops::Range<usize>>)> {
    let n_pairs: u32 = pairs
        .len()
        .try_into()
        .map_err(|_| ScalibError::TooManyPois)?;
    let mut pairs_matrix = ndarray::Array2::from_elem((ns, ns), vec![]);
    for (k, (i, j)) in pairs.iter().enumerate() {
        let (i, j) = (*i.min(j), *i.max(j));
        pairs_matrix[(i as usize, j as usize)].push(k as u32);
    }
    let cw = max_chunk_size / 2;
    let mut pair_to_old_idx: Vec<u32> = vec![];
    let mut chunks = vec![];
    for i_chunk in 0..ns.div_ceil(cw) {
        let i_start = i_chunk * cw;
        let i_end = std::cmp::min((i_chunk + 1) * cw, ns);
        let mut chunk_size = 0;
        let mut chunk_start = pair_to_old_idx.len();
        for j in i_start..ns {
            let mut empty = true;
            pair_to_old_idx.extend(
                (i_start..i_end)
                    .flat_map(|i| pairs_matrix[(i, j)].iter())
                    .inspect(|_| {
                        empty = false;
                    }),
            );
            if !empty {
                chunk_size += 1;
                if chunk_size == cw {
                    chunk_size = 0;
                    chunks.push(chunk_start..pair_to_old_idx.len());
                    chunk_start = pair_to_old_idx.len();
                }
            }
        }
        if chunk_size != 0 {
            chunks.push(chunk_start..pair_to_old_idx.len());
        }
    }
    debug_assert!(itertools::equal(
        pair_to_old_idx.iter().copied().sorted_unstable(),
        0..n_pairs
    ));
    assert_eq!(chunks[0].start, 0);
    assert_eq!(chunks.last().unwrap().end, pair_to_old_idx.len());
    assert!(chunks
        .iter()
        .zip(chunks[1..].iter())
        .all(|(c, d)| c.end == d.start));
    Ok((pair_to_old_idx, chunks))
}

fn multi_split_at<T>(mut slice: &[T], split_at: impl Iterator<Item = usize>) -> Vec<&[T]> {
    let mut last = 0;
    let mut res = Vec::with_capacity(split_at.size_hint().0);
    for x in split_at {
        let (s, rem) = slice.split_at(x - last);
        slice = rem;
        last = x;
        res.push(s);
    }
    res.push(slice);
    res
}
fn multi_split_at_mut<T>(
    mut slice: &mut [T],
    split_at: impl Iterator<Item = usize>,
) -> Vec<&mut [T]> {
    let mut last = 0;
    let mut res = Vec::with_capacity(split_at.size_hint().0);
    for x in split_at {
        let (s, rem) = slice.split_at_mut(x - last);
        slice = rem;
        last = x;
        res.push(s);
    }
    res.push(slice);
    res
}
