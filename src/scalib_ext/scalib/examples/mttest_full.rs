use ndarray::array;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;

fn main() {
    let d = 2;
    let ns = 10000;
    let npois = 5000000;
    let n = 10;
    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));
    let pois = Array2::<u64>::random((d, npois), Uniform::new(0, ns as u64));
    let mut mtt = ttest::MTtest::new(d, pois.view());
    mtt.update(traces.view(), y.view());
}
