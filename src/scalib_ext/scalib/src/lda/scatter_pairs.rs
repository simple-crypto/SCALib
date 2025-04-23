/// Compute sparse scatter matrix.
///
/// Given a list of pairs `(i, j)` such that `i <= j` and a set of traces, compute the scatter
/// `sum_{trace in traces} trace[i]*traces[j]`.
use itertools::izip;
use ndarray::{Array2, ArrayView2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{Result, ScalibError};

use super::poi_map::{PoiMap, AA};

// Let ns be the length of the traces, the algorithm works by partitioning the `ns*ns` scatter
// matrix into rectangular blocks. We iterate in parallel over the blocks to update them.
// Each block update iterate over chunks of traces, then over the pairs of POIs in the block and
// finally over the traces in the block.
// This last loop is implemented using SIMD (pairwise sum of i16 products, then parallel SIMD sum
// and finally SIMD reduction in a single sum).
//
// The parameters are:
// - `N`: number of traces in a chunk
// - `MAX_PAIRS_BLOCK`: maximum number of computed POI pairs in a block
// - `MAX_POIS_BLOCK`: maximum number of POIs in the pairs a single block.
//
// The memory usage is:
// - indices of POIs in the pairs: at most `8*MAX_PAIRS_BLOCK` bytes per block.
// - scatter accumulator: at most `8*MAX_PAIRS_BLOCK` bytes per block.
// - traces: at most `2*N*MAX_POIS_BLOCK` bytes per block.
//
// The indices and scatter accumulator have a somewhat low read/write throughput, so they can be in
// L3 cache. Ideally, we would assume 2MB L3 per core and use half of it for these two values.
// In order to have a good efficiency in the innermost loop, the horizontal reduction should be
// outweighted by a large enough number of iterations. One AVX2 iteration processes 16 traces with
// 2 arithmetic instructions (horizontal-adding multiply + addition), so N=64 is a minimum.
// Traces is `2*N` bytes per POI and should ideally fit in half L2 cache (high throughput), and
// `N >= 64` for decent compute efficiency.
// Assuming 512 kB L2 cache, that gives `MAX_POIS_BLOCK=2**11`.
// For streaming the traces from memory, we use `2*N` bytes of bandwith per POI, for a
// compute of ~12 arithmetic instructions (~4 clock cycles) per pair at N=64.
// Assuming 4 GHz clock, that gives `2*N n_pois/n_pairs` B/s. Assuming we can stream 1GB/s per core,
// we need `n_pairs >= 128*n_pois`.
// If we compute a fully-dense covariance, `n_pairs = n_pois**2/4`, this means `n_pois >= 2**9`.
// In practice, we want to handle efficiently sparser matrices, so we need a larger
// `MAX_POIS_BLOCK`. With `2**11`, we can have 25% of density without being memory-bottlenecked.
// To achieve this, we would have `MAX_PAIRS_BLOCK = 1/4 * (MAX_POIS_BLOCK/2)**2 = 2**18`,
// which leads to `2**22` bytes (4MB) of L3 usage per core. That's a bit too much, so we reduce it
// down to `MAX_PAIRS_BLOCK=2**17` and accept a bit of loss in not-so-dense cases.
// We also scale down MAX_POI_BLOCKS to 2**10, so in
// "half bad" cases, we can actually reach optimal efficiency.
// TODO: make this adaptative ?
// TODO: make sample indices u16 instead of u32 when possible.
// TODO: revisit values ?
const MAX_PAIRS_BLOCK: usize = 1 << 17;
const MAX_POIS_BLOCK: usize = 1 << 10;
const N: usize = 64;

/// Sparse scatter accumulator.
// TODO: if ns is too large, use roaring bitset for pairs_to_new_idx
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPairs {
    /// Pairs are pairs of POIs for which the covariance is computed.
    /// Values in `sorted_pairs` are sorted such that consecutive pairs have good memory locality
    /// (i.e., tend to have the same POIs).
    sorted_pairs: Vec<(u32, u32)>,
    /// Ranges in `sorted_pairs` corresponding to values that should be computed together for
    /// good cache locality.
    blocks: Vec<std::ops::Range<usize>>,
    /// Mapping between orignal pair indices and the corresponding pair in `sorted_pairs`.
    pub pairs_to_new_idx: Array2<u32>,
    /// Number of traces accumulated.
    pub tot_n_traces: u32,
    /// Scatter values, in the same order as sorted_pairs
    /// (sums of products of samples).
    pub scatter: Vec<i64>,
}

impl ScatterPairs {
    /// ns: length of the traces
    /// pairs: list of (i, j) pairs
    pub fn new(ns: usize, pairs: impl Iterator<Item = (u32, u32)>) -> Result<Self> {
        let (sorted_pairs, pairs_to_new_idx, blocks) = blocks_of_pairs(pairs, ns)?;
        Ok(Self {
            scatter: vec![0; sorted_pairs.len()],
            sorted_pairs,
            blocks,
            pairs_to_new_idx,
            tot_n_traces: 0,
        })
    }

    /// traces shape: (ntraces, ns)
    pub fn update(&mut self, poi_map: &PoiMap, traces: ArrayView2<i16>) -> Result<()> {
        assert!(traces.is_standard_layout());
        let n_traces: u32 = traces.shape()[0]
            .try_into()
            .map_err(|_| ScalibError::TooManyTraces)?;
        self.tot_n_traces = self
            .tot_n_traces
            .checked_add(n_traces)
            .ok_or(ScalibError::TooManyTraces)?;
        let traces = poi_map.select_batches::<N>(traces);
        let split_at = self.blocks[1..].iter().map(|r| r.start);
        let scatter_blocks = multi_split_at_mut(self.scatter.as_mut_slice(), split_at.clone());
        let pair_blocks = multi_split_at(self.sorted_pairs.as_slice(), split_at);
        (scatter_blocks, pair_blocks)
            .into_par_iter()
            .for_each(|(scatter_block, pair_block)| {
                Self::update_block(&traces, scatter_block, pair_block);
            });
        Ok(())
    }

    /// traces: see PoiMap::select_batches
    /// scatter_block and pair_block are matching slices of scatter and
    /// sorted_pairs.
    fn update_block(traces: &Array2<AA<N>>, scatter_block: &mut [i64], pair_block: &[(u32, u32)]) {
        assert!(traces.is_standard_layout());
        for batch in traces.outer_iter() {
            for (scatter, (i, j)) in izip!(scatter_block.iter_mut(), pair_block.iter()) {
                *scatter += sum_prod(&batch[*i as usize], &batch[*j as usize]);
            }
        }
    }

    pub(crate) fn get_scatter(&self, i: u32, j: u32) -> i64 {
        self.scatter[self.pairs_to_new_idx[(i as usize, j as usize)] as usize]
    }
}

/// Compute the dot product of the vectors x and y
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn sum_prod<'a>(x: &'a AA<N>, y: &'a AA<N>) -> i64 {
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
}

/// Compute the dot product of the vectors x and y (default, non-intrinsics)
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn sum_prod<'a>(x: &'a AA<N>, y: &'a AA<N>) -> i64 {
    let mut res = 0;
    for k in 0..N {
        res += ((x.0[k] as i32) * (y.0[k] as i32)) as i64;
    }
    res
}

/// Compute the block partitioning of the pairs.
/// - pairs: unsorted, possibly redudant list of pairs
/// - ns: trace length
/// Returns:
/// - a Vec of pairs
/// - a map of pairs to indices in the previous Vec (as an Array2)
/// - a list of ranges in the Vec, corresponding to blocks.
fn blocks_of_pairs(
    pairs: impl Iterator<Item = (u32, u32)>,
    ns: usize,
) -> Result<(Vec<(u32, u32)>, Array2<u32>, Vec<std::ops::Range<usize>>)> {
    let mut pairs_matrix = Array2::from_elem((ns, ns), false);
    for (i, j) in pairs {
        pairs_matrix[(i as usize, j as usize)] = true;
    }
    let n_pairs = pairs_matrix.iter().filter(|x| **x).count();
    assert!(n_pairs <= i32::MAX as usize);
    let _: i32 = n_pairs.try_into().map_err(|_| ScalibError::TooManyPois)?;
    let cw = MAX_POIS_BLOCK / 2;
    let mut pairs_to_new_idx = Array2::from_elem((ns, ns), u32::MAX);
    let mut sorted_pairs = vec![];
    let mut blocks = vec![];
    // TODO: extend i_block len when there is only a single j block
    for i_block_id in 0..ns.div_ceil(cw) {
        let i_start = i_block_id * cw;
        let i_end = std::cmp::min((i_block_id + 1) * cw, ns);
        let i_block = i_start..i_end;
        let mut block_start = sorted_pairs.len();
        let mut used_i = vec![false; i_end - i_start];
        let mut block_poi_used = 0;
        for j in i_start..ns {
            // Check if we can add this j without making the block too big.
            let mut n_new_poi_pairs = 0;
            let mut n_newly_used_i = 0;
            let mut j_is_a_new_i = false;
            for i in i_block.clone() {
                if pairs_matrix[(i, j)] {
                    n_new_poi_pairs += 1;
                    if !used_i[i - i_start] {
                        n_newly_used_i += 1;
                        if i == j {
                            j_is_a_new_i = true;
                        }
                    }
                }
            }
            let j_reused = j_is_a_new_i || (i_block.contains(&j) && used_i[j - i_start]);
            let n_newly_used = n_newly_used_i + if j_reused { 0 } else { 1 };
            // If not possible, start a new block.
            if block_poi_used + n_newly_used > MAX_POIS_BLOCK
                || (sorted_pairs.len() - block_start) + n_new_poi_pairs > MAX_PAIRS_BLOCK
            {
                blocks.push(block_start..sorted_pairs.len());
                block_start = sorted_pairs.len();
                used_i.fill(false);
                block_poi_used = 0;
            }
            // Add all pairs to the current block.
            let mut used_j = false;
            for i in i_block.clone() {
                if pairs_matrix[(i, j)] {
                    pairs_to_new_idx[(i, j)] = sorted_pairs.len() as u32;
                    sorted_pairs.push((i as u32, j as u32));
                    if !used_i[i - i_start] {
                        used_i[i - i_start] = true;
                        block_poi_used += 1;
                    }
                    if !used_j {
                        if i_block.contains(&j) {
                            if !used_i[j - i_start] {
                                used_i[j - i_start] = true;
                                block_poi_used += 1;
                            }
                        } else {
                            block_poi_used += 1;
                        }
                    }
                    used_j = true;
                }
            }
        }
        if block_start != sorted_pairs.len() {
            blocks.push(block_start..sorted_pairs.len());
        }
    }
    assert!(blocks
        .iter()
        .zip(blocks[1..].iter())
        .all(|(c, d)| c.end == d.start));
    Ok((sorted_pairs, pairs_to_new_idx, blocks))
}

/// Split slice according to multiple split points.
/// If split_at=[i_0, i_1, ..., i_n], then return
/// return vec![&slice[0..i_0], &slice[i_0..i_1], ..., &slice[i_{n-1},i_n]]
/// Note that there is no &slice[i_n..] !.
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

/// See `multi_split_at`.
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
