use ndarray::array;
use ndarray::{s, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::mttest;

fn main() {
    let d = 2;
    let ns = 10;
    let n = 20;
    let npois = 10;
    let traces = Array2::<i16>::random((n, ns as usize), Uniform::new(0, ns));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));
    let pois = Array2::<u32>::random((2, npois), Uniform::new(0, ns as u32));
    let mut ttacc = mttest::MTtest::new(2,pois.view());
    ttacc.update(traces.view(), y.view());
    let t = ttacc.get_ttest();
    println!("ttest {:#?}",t);
}
