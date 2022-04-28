use ndarray::array;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;

fn main() {
    let d = 2;
    let ns = 100;
    let n = 2;
    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));
    let mut ttacc = ttest::TtestAcc::new(ns as usize, d);
    ttacc.update(traces.view(), y.view());
}
