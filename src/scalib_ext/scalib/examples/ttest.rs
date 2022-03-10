use ndarray::array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;

fn main() {
    println!("Welcome to ttest example");

    let d = 2;
    let ns = 1000;
    let pois = array![[0, 1, 2], [0, 2, 1]];
    let traces = array![[0, 1, 2], [2, 2, 2], [1, 1, 0]];
    let y = array![0, 1, 1];

    let mut ttest = ttest::MTtest2::new(pois);
    ttest.update(traces.view(), y.view());
    ttest.get_ttest();
}
