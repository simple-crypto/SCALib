use super::poi_map::PoiMap;
use ndarray::{Array2, ArrayView2, Axis};

// TODO: have indices be i16 when possible.
#[derive(Debug, Clone)]
pub struct BatchedTraces<'a, 'b, const N: usize> {
    poi_map: &'b PoiMap,
    traces: ArrayView2<'a, i16>,
}

#[derive(Debug, Clone)]
#[repr(C, align(32))]
pub struct AA<const N: usize>(pub [i16; N]);

impl<const N: usize> AA<N> {
    fn sum(&self) -> i64 {
        self.0.iter().map(|x| *x as i64).sum()
    }
}

impl<'a, 'b, const N: usize> BatchedTraces<'a, 'b, N> {
    pub fn new(poi_map: &'b PoiMap, traces: ArrayView2<'a, i16>) -> Self {
        Self { poi_map, traces }
    }
    fn map<'s, T, F>(&'s self, mut f: F) -> impl Iterator<Item = T> + 's
    where
        F: FnMut(u32, &[AA<N>]) -> T + 's,
        'a: 's,
        'b: 's,
    {
        let mut batch: Vec<AA<N>> = vec![AA([0; N]); self.traces.shape()[1]];
        let kept_indices: &'s _ = self.poi_map.kept_indices();
        self.traces.axis_chunks_iter(Axis(0), N).map(move |chunk| {
            transpose_big(chunk, batch.as_mut_slice(), kept_indices);
            f(chunk.shape()[0] as u32, batch.as_slice())
        })
    }
    fn for_each<'s, F>(&'s self, f: F)
    where
        F: FnMut(u32, &[AA<N>]) + 's,
        'a: 's,
    {
        self.map(f).for_each(|_| {})
    }
    fn try_for_each<'s, F, E>(&'s self, f: F) -> std::result::Result<(), E>
    where
        F: FnMut(u32, &[AA<N>]) -> std::result::Result<(), E> + 's,
        'a: 's,
    {
        self.map(f).try_for_each(|x| x)
    }
    // TODO: make a parallel version of this.
    #[inline(never)]
    pub fn get_batches(&self) -> Array2<AA<N>> {
        let n_batches = self.traces.shape()[0].div_ceil(N);
        let mut res = Array2::from_elem((n_batches, self.traces.shape()[1]), AA([0; N]));
        let chunk_iter = self.traces.axis_chunks_iter(Axis(0), N);
        for (chunk, mut batch) in chunk_iter.zip(res.outer_iter_mut()) {
            transpose_big(
                chunk,
                batch.as_slice_mut().unwrap(),
                self.poi_map.kept_indices(),
            );
        }
        res
    }
}

#[inline(never)]
fn transpose_big<const N: usize>(
    chunk: ArrayView2<i16>,
    batch: &mut [AA<N>],
    kept_indices: &[u32],
) {
    assert!(chunk.shape()[0] <= N);
    assert_eq!(chunk.shape()[1], batch.len());
    if chunk.shape()[0] != N {
        batch.fill(AA([0; N]));
    }
    const SUB_CHUNK_M: usize = 8;
    const SUB_CHUNK_N: usize = 8;
    for (sb_i, traces_sb) in chunk.axis_chunks_iter(Axis(0), SUB_CHUNK_M).enumerate() {
        for (indices_sb, chunk_sb) in kept_indices
            .chunks(SUB_CHUNK_N)
            .zip(batch.chunks_mut(SUB_CHUNK_N))
        {
            if indices_sb.len() == SUB_CHUNK_N && traces_sb.shape()[0] == SUB_CHUNK_M {
                let mut chunk_sb_iter = chunk_sb.iter_mut();
                let y: [&mut [i16; SUB_CHUNK_M]; SUB_CHUNK_N] = std::array::from_fn(|i| {
                    let offset = sb_i * SUB_CHUNK_M;
                    let sl = &mut chunk_sb_iter.next().unwrap().0[offset..(offset + SUB_CHUNK_M)];
                    sl.try_into().unwrap()
                });
                transpose_small(traces_sb, indices_sb, y);
            } else {
                for (idx, batch_item) in indices_sb.iter().zip(chunk_sb.iter_mut()) {
                    for i in 0..traces_sb.shape()[0] {
                        batch_item.0[sb_i * SUB_CHUNK_M + i] = traces_sb[(i, *idx as usize)];
                    }
                }
            }
        }
    }
}

// TODO: maybe there can be a bit of SIMD opt here...
// basic: load i16 and insert in SIMD, write-back as full word.
// improved: detect when indices are consecutive, and go more SIMD there (maybe 64-bit is enough ?)
// https://stackoverflow.com/questions/2517584/transpose-for-8-registers-of-16-bit-elements-on-sse2-ssse3/2518670#2518670
#[inline(never)]
fn transpose_small<const M: usize, const N: usize>(
    x: ArrayView2<i16>,
    indices: &[u32],
    y: [&mut [i16; M]; N],
) {
    assert_eq!(x.shape()[0], M);
    assert_eq!(indices.len(), N);
    for j in 0..N {
        for i in 0..M {
            y[j][i] = x[(i, indices[j] as usize)];
        }
    }
}
