//! Computation of low-moment leakage statistics: per-class means and global variance for analysis
//! of the leakage's variance decomposition.
//!
//! The statistics can be computed for nv independent random variables, for traces of ns samples.
//! The class variable is in the range [0,nc[.

use crate::ScalibError;
use hytra::TrAdder;
use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayViewMut2, Axis, Zip};
use num_traits::{Bounded, PrimInt, Signed, WrappingAdd, Zero};
use rayon::prelude::*;
use std::convert::TryInto;

pub trait NativeInt: PrimInt + Signed + WrappingAdd + Send + Sync + std::fmt::Debug {}
impl<T: PrimInt + Signed + WrappingAdd + Send + Sync + std::fmt::Debug> NativeInt for T {}

pub trait AccType: std::fmt::Debug {
    type SumAcc: NativeInt;
    fn tmp2acc(s: i32) -> Self::SumAcc;
    fn acc2i64(acc: Self::SumAcc) -> i64;
}

const AVX2_SIZE: usize = 32;
const SMALL_ACC_BYTES: usize = 4; // i32
pub const SIMD_SIZE: usize = AVX2_SIZE / SMALL_ACC_BYTES;
const CACHE_LINE: usize = 64;
const TRACE_SAMPLE_BYTES: usize = 2; // i16
const SAMPLES_BLOCK_SIZE: usize = CACHE_LINE / TRACE_SAMPLE_BYTES;
const TRACES_CHUNK_SIZE: usize = 1024;

#[derive(Debug)]
pub struct AccType64bit;
#[derive(Debug)]
pub struct AccType32bit;

impl AccType for AccType64bit {
    type SumAcc = i64;
    #[inline(always)]
    fn tmp2acc(s: i32) -> Self::SumAcc {
        s as i64
    }
    #[inline(always)]
    fn acc2i64(acc: Self::SumAcc) -> i64 {
        acc
    }
}
impl AccType for AccType32bit {
    type SumAcc = i32;
    #[inline(always)]
    fn tmp2acc(s: i32) -> Self::SumAcc {
        s as i32
    }
    #[inline(always)]
    fn acc2i64(acc: Self::SumAcc) -> i64 {
        acc as i64
    }
}

/// LVar state. stores the sum and the sum of squares of the leakage for each of the class.
#[derive(Debug)]
pub struct LVar<T: AccType> {
    /// Sum of all the traces corresponding to each of the classes. shape (ceil(ns/SIMD_SIZE),nv,nc)
    sum: Array3<[T::SumAcc; SIMD_SIZE]>,
    /// Sum of squares with shape (ceil(ns/SIMD_SIZE))
    /// (never overflows since samples are i16 and tot_n_samples <= u32::MAX)
    sum_square: Array1<[i64; SIMD_SIZE]>,
    /// number of samples per class (nv,nc)
    n_samples: Array2<u32>,
    /// number of independent variables
    nv: usize,
    /// number of samples in a trace
    ns: usize,
    /// number of classes
    nc: u32,
    /// max sample bit width
    bit_width: u32,
    /// total number of accumulated traces
    tot_n_samples: u32,
}

impl<T: AccType> LVar<T> {
    /// Create a new LVar state.
    /// nc: random variables between [0,nc[
    /// ns: traces length
    /// nv: number of independent random variables
    pub fn new(nc: usize, ns: usize, nv: usize) -> Self {
        let ns_b = ns.div_ceil(SIMD_SIZE);
        assert!(nc <= 1 << 16);
        LVar {
            sum: Array3::from_elem((ns_b, nv, nc), [Zero::zero(); SIMD_SIZE]),
            sum_square: Array1::from_elem((ns_b,), [0; SIMD_SIZE]),
            n_samples: Array2::zeros((nv, nc)),
            nv,
            ns,
            nc: nc.try_into().expect("Too many classes"),
            bit_width: 1,
            tot_n_samples: 0,
        }
    }

    /// Update the LVar state with n fresh traces
    /// traces: the leakage traces with shape (n,ns)
    /// y: realization of random variables with shape (nv,n)
    /// If this errors, the SNR object should not be used anymore.
    /// traces and y must be in standard C order
    pub fn update(
        &mut self,
        traces: ArrayView2<i16>,
        y: ArrayView2<u16>,
        config: &crate::Config,
    ) -> Result<(), ScalibError> {
        let n_it = (self.sum.shape()[0] as u64 + 3) / 4;
        crate::utils::with_progress(
            |it_cnt| self.update_internal(traces, y, it_cnt),
            n_it,
            "Update SNR",
            config,
        )
    }

    #[inline(never)]
    /// If this errors, the SNR object should not be used anymore.
    fn update_internal(
        &mut self,
        // shape (n, ns)
        traces: ArrayView2<i16>,
        // shape (nv, n)
        y: ArrayView2<u16>,
        acc_ref: &TrAdder<u64>,
    ) -> Result<(), ScalibError> {
        assert_eq!(traces.shape()[0], y.shape()[1]);
        assert_eq!(traces.shape()[1], self.ns);
        assert_eq!(y.shape()[0], self.nv);
        assert!(traces.is_standard_layout());
        assert!(y.is_standard_layout());
        let n_traces: u32 = traces.shape()[0]
            .try_into()
            .map_err(|_| ScalibError::TooManyTraces)?;
        self.tot_n_samples = self
            .tot_n_samples
            .checked_add(n_traces)
            .ok_or(ScalibError::TooManyTraces)?;
        let mut max_n_samples: u32 = 0;
        let nc = self.nc;
        let nv = self.nv;
        izip!(self.n_samples.outer_iter_mut(), y.outer_iter()).try_for_each(
            |(mut n_samples, y)| {
                y.into_iter().try_for_each(|y| {
                    if u32::from(*y) >= nc {
                        Err(ScalibError::SnrClassOutOfBound)
                    } else {
                        n_samples[*y as usize] += 1;
                        max_n_samples = std::cmp::max(max_n_samples, n_samples[*y as usize]);
                        Ok(())
                    }
                })
            },
        )?;
        let sample_bits_used_msk = (
            self.sum
                .axis_chunks_iter_mut(Axis(0), SAMPLES_BLOCK_SIZE / SIMD_SIZE),
            self.sum_square
                .axis_chunks_iter_mut(Axis(0), SAMPLES_BLOCK_SIZE / SIMD_SIZE),
            traces.axis_chunks_iter(Axis(1), SAMPLES_BLOCK_SIZE),
        )
            .into_par_iter()
            .map_init(
                || {
                    (
                        Array2::from_elem((4, TRACES_CHUNK_SIZE), [0i16; SIMD_SIZE]),
                        Array3::from_elem((4, nv, nc as usize), [0i32; SIMD_SIZE]),
                    )
                },
                |(traces_tr, tmp_sum), (mut sum, mut sum_square, trace_chunk)| {
                    let mut sample_bits_used_msk = 0;
                    izip!(
                        trace_chunk.axis_chunks_iter(Axis(0), u16::MAX as usize),
                        y.axis_chunks_iter(Axis(1), u16::MAX as usize)
                    )
                    .for_each(|(trace_chunk, y)| {
                        tmp_sum.fill([0; SIMD_SIZE]);
                        izip!(
                            trace_chunk.axis_chunks_iter(Axis(0), TRACES_CHUNK_SIZE),
                            y.axis_chunks_iter(Axis(1), TRACES_CHUNK_SIZE)
                        )
                        .for_each(|(trace_chunk, y)| {
                            let mut traces_tr =
                                traces_tr.slice_mut(s![.., ..trace_chunk.shape()[0]]);
                            sample_bits_used_msk |=
                                transpose_traces(traces_tr.view_mut(), trace_chunk);
                            izip!(
                                traces_tr.axis_iter(Axis(0)),
                                tmp_sum.axis_iter_mut(Axis(0)),
                                sum_square.axis_iter_mut(Axis(0)),
                            )
                            .for_each(
                                |(traces_chunk, sum, sum_square)| {
                                    let traces_chunk = traces_chunk.to_slice().unwrap();
                                    // SAFETY: y has been checked before, and offset/stride is in bound
                                    unsafe {
                                        inner_lvar_update(
                                            traces_chunk,
                                            y,
                                            sum,
                                            sum_square.into_scalar(),
                                        );
                                    }
                                },
                            );
                        });
                        for (mut sum, tmp_sum) in
                            izip!(sum.axis_iter_mut(Axis(0)), tmp_sum.axis_iter(Axis(0)))
                        {
                            Zip::from(&mut sum).and(tmp_sum).for_each(|sum, tmp_sum| {
                                for (sum, tmp_sum) in sum.iter_mut().zip(tmp_sum.iter()) {
                                    *sum = sum.wrapping_add(&T::tmp2acc(*tmp_sum));
                                }
                            });
                        }
                    });
                    acc_ref.inc(1);
                    sample_bits_used_msk
                },
            )
            .reduce(|| 0, |a, b| a | b);
        self.bit_width = std::cmp::max(self.bit_width, 16 - sample_bits_used_msk.leading_zeros());
        // for any sample x, abs(x) < 2^bit_width
        // we want max_n_samples*abs(x) < T::SumAcc::max_value(), therefore
        // max_n_samples*abs(x) << bit_width \le T::SumAcc::max_value()
        // max_val does not overflow since max_n_samples < 2^32 and self.bit_width < 16
        let max_val = (max_n_samples as i64) << self.bit_width;
        if max_val > T::acc2i64(T::SumAcc::max_value()) {
            return Err(ScalibError::SnrClassOverflow {
                leak_upper_bound: 1 << self.bit_width,
                max_n_traces: max_n_samples as i64,
            });
        }
        return Ok(());
    }

    pub fn tot_sum(&self) -> Array1<[i64; SIMD_SIZE]> {
        self.sum
            .index_axis(Axis(1), 0)
            .axis_iter(Axis(0))
            .map(|sums| {
                sums.iter().fold([0; SIMD_SIZE], |x, y| {
                    std::array::from_fn(|i| x[i] + T::acc2i64(y[i]))
                })
            })
            .collect::<Vec<_>>()
            .into()
    }

    pub fn sum(&self) -> &Array3<[T::SumAcc; SIMD_SIZE]> {
        &self.sum
    }
    pub fn sum_square(&self) -> &Array1<[i64; SIMD_SIZE]> {
        &self.sum_square
    }
    pub fn n_samples(&self) -> &Array2<u32> {
        &self.n_samples
    }
    pub fn tot_n_samples(&self) -> u32 {
        self.tot_n_samples
    }
    pub fn nv(&self) -> usize {
        self.nv
    }
    pub fn ns(&self) -> usize {
        self.ns
    }
    pub fn nc(&self) -> usize {
        self.nc as usize
    }
}

#[inline(never)]
///  # Safety
///  all values in y must be < sum.shape()[1]
unsafe fn inner_lvar_update(
    // len: n
    trace_chunk: &[[i16; SIMD_SIZE]],
    // (nv, n)
    y: ArrayView2<u16>,
    // (nv, nc)
    mut sum: ArrayViewMut2<[i32; SIMD_SIZE]>,
    sum_square: &mut [i64; SIMD_SIZE],
) {
    assert_eq!(trace_chunk.len(), y.shape()[1]);
    assert_eq!(sum.shape()[0], y.shape()[0]);
    for trace in trace_chunk {
        for (sum_square, trace) in sum_square.iter_mut().zip(trace.iter()) {
            let trace = *trace as i64;
            // overflow handled with error elsewhere
            *sum_square = sum_square.wrapping_add(trace * trace);
        }
    }
    izip!(y.outer_iter(), sum.outer_iter_mut()).for_each(|(y, sum)| {
        let sum = sum.into_slice().unwrap();
        izip!(y.to_slice().unwrap(), trace_chunk).for_each(|(y, trace_chunk)| {
            // sum.get_unchecked_mut is safe due to assumption,
            let sum = sum.get_unchecked_mut(*y as usize);
            for j in 0..SIMD_SIZE {
                // overflow handled with error elsewhere
                sum[j] = sum[j].wrapping_add(trace_chunk[j] as i32);
            }
        })
    });
}

#[inline(never)]
fn transpose_traces(
    // shape: (4, n)
    mut traces_tr: ArrayViewMut2<[i16; SIMD_SIZE]>,
    // shape: (n, ns) with ns <= SAMPLES_BLOCK_SIZE
    trace_chunk: ArrayView2<i16>,
) -> u16 {
    assert_eq!(traces_tr.shape()[1], trace_chunk.shape()[0]);
    assert_eq!(traces_tr.shape()[0], 4);
    assert!(trace_chunk.shape()[1] <= SAMPLES_BLOCK_SIZE);
    let mut max_width: u16 = 0;
    if trace_chunk.shape()[1] == SAMPLES_BLOCK_SIZE {
        let mut max_width_vec = [0u16; SIMD_SIZE];
        izip!(
            traces_tr.axis_iter_mut(Axis(1)),
            trace_chunk.axis_iter(Axis(0))
        )
        .for_each(|(mut traces_tr, trace_chunk)| {
            izip!(
                traces_tr.iter_mut(),
                trace_chunk.axis_chunks_iter(Axis(0), SIMD_SIZE)
            )
            .for_each(|(traces_tr, trace_chunk)| {
                let trace_chunk: &[i16; SIMD_SIZE] =
                    trace_chunk.to_slice().unwrap().try_into().unwrap();
                //traces_tr.clone_from_slice(trace_chunk);
                *traces_tr = *trace_chunk;
                for (max_width, trace_chunk) in max_width_vec.iter_mut().zip(trace_chunk.iter()) {
                    // i16::abs_diff returns a u16 without overflow nor panic, while i16::abs
                    // panics on i16::min_value() input.
                    *max_width |= trace_chunk.abs_diff(0);
                }
            });
        });
        for mw in max_width_vec {
            max_width |= mw;
        }
    } else {
        izip!(
            traces_tr.axis_iter_mut(Axis(1)),
            trace_chunk.axis_iter(Axis(0))
        )
        .for_each(|(mut traces_tr, trace_chunk)| {
            izip!(
                traces_tr.iter_mut().flat_map(|x| x.iter_mut()),
                trace_chunk.iter()
            )
            .for_each(|(traces_tr, trace_chunk)| {
                *traces_tr = *trace_chunk;
                max_width |= trace_chunk.abs_diff(0);
            });
        });
    }
    return max_width;
}
