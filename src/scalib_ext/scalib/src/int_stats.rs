use num_traits::{PrimInt, Signed, WrappingAdd, Zero};
pub trait NativeInt: PrimInt + Signed + WrappingAdd + Send + Sync {}
impl<T: PrimInt + Signed + WrappingAdd + Send + Sync> NativeInt for T {}

use crate::{Result, ScalibError};
use ndarray::{azip, s, Array1, Array2, ArrayView2, Axis};

/// Each array represnets N consecutive traces.
#[derive(Debug, Clone)]
pub struct BatchedTraces<const N: usize> {
    batches: Array2<[i16; N]>,
    rem: Array2<i16>,
}

impl<const N: usize> BatchedTraces<N> {
    pub fn new(traces: ArrayView2<i16>) -> Self {
        let n_traces = traces.shape()[0];
        let n_batched_traces = n_traces & !(N - 1);
        let rem = traces.slice(s![n_batched_traces.., ..]).to_owned();

        let batches = azip!(traces.exact_chunks((N, 1)))
            .map_collect(|col| std::array::from_fn(|i| col[(i, 0)]));
        Self { batches, rem }
    }
}

#[derive(Debug, Clone)]
pub struct CovAcc<T: NativeInt> {
    pub tot_n_traces: u32,
    pub scatter: Array2<T>,
    pub sums: Array1<T>,
    pub ns: usize,
}

//impl<T: NativeInt> CovAcc<T> {
impl CovAcc<i64> {
    pub fn from_dim(ns: usize) -> Self {
        Self {
            ns,
            tot_n_traces: 0,
            scatter: Array2::zeros((ns, ns)),
            sums: Array1::zeros((ns,)),
        }
    }
    pub fn update(&mut self, traces: ArrayView2<i16>) -> Result<()> {
        assert_eq!(traces.shape()[1], self.ns);
        assert!(traces.is_standard_layout());
        let n_traces: u32 = traces.shape()[0]
            .try_into()
            .map_err(|_| ScalibError::TooManyTraces)?;
        self.tot_n_traces = self
            .tot_n_traces
            .checked_add(n_traces)
            .ok_or(ScalibError::TooManyTraces)?;
        let traces = BatchedTraces::<16>::new(traces);
        // TODO: assert no overflow
        for batch in traces.batches.axis_iter(Axis(0)) {
            for i in 0..self.ns {
                for j in i..self.ns {
                    self.scatter[(i, j)] = self.scatter[(i, j)]
                        + batch[i]
                            .iter()
                            .zip(batch[j].iter())
                            .map(|(i, j)| ((*i as i32) * (*j as i32)) as i64)
                            .sum::<i64>();
                }
            }
        }
        Ok(())
    }
}
