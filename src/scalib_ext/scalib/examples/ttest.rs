use ndarray::array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;

fn main() {
    println!("Welcome to ttest example");

    let d = 2;
    let ns = 3;
    //let pois = array![[0, 1, 2], [0, 1, 2], [0, 1, 2]];
    let pois = array![[0, 1, 2], [0, 1, 2]];
    let traces = array![
        [-1, -1, -1],
        [1, 1, 1],
        [-1, -1, -1],
        [-2, -2, -2],
        [2, 2, 2],
        [-2, -2, -2]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    let mut ttest2 = ttest::MTtest::new(d, pois.view());
    ttest2.update(traces.view(), y.view());
    let re2 = ttest2.get_ttest();
    println!("ttest2 output {}", re2);

    let mut ttest = ttest::Ttest::new(ns, d);
    ttest.update(traces.view(), y.view());
    let re = ttest.get_ttest();
    println!("ttest output {}", re);
}
