use ndarray::{array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;

fn main() {
    println!("Welcome to ttest example");

    let d = 2;
    let ns = 3;
    let pois = array![[0, 1, 2], [0, 1, 2]];

    let traces = Array2::<i16>::random((100, 5), Uniform::<i16>::new(0, 10));
    let y = Array1::<u16>::random((100,), Uniform::<u16>::new(0, 2));

    let mut ttest2 = ttest::MTtest::new(d, pois.view());
    ttest2.update(traces.view(), y.view());
    let re2 = ttest2.get_ttest();
    println!("ttest2 output {}", re2);

    let mut ttest = ttest::Ttest::new(ns, d);
    ttest.update(traces.view(), y.view());
    let re = ttest.get_ttest();
    println!("ttest output {}", re);
}
