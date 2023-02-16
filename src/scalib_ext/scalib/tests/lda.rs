use ndarray::{Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256StarStar;
use scalib::lda;

fn gen_problem(n: usize, ns: usize, nc: u16, x_lb: i16, x_ub: i16) -> (Array2<i16>, Array1<u16>) {
    let seed = 42;
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let x = Array2::<i16>::random_using((n, ns), Uniform::new(x_lb, x_ub), &mut rng);
    let y = Array1::<u16>::random_using(n, Uniform::new(0, nc), &mut rng);
    return (x, y);
}

#[test]
fn fit_lda() {
    let ns = 100;
    let n = 1000;
    let nc = 16;

    let (x, y) = gen_problem(n, ns, nc, 0, 100);
    let _lda = lda::LdaAcc::new(nc.into(), x.view(), y.view(), 0);
}

#[test]
fn predict_lda() {
    let ns = 100;
    let n = 1000;
    let nc = 16;
    let p = 4;

    let (x, y) = gen_problem(n, ns, nc, 0, 100);
    let lda = lda::LdaAcc::new(nc.into(), x.view(), y.view(), 0)
        .lda(p)
        .unwrap();

    lda.predict_proba(x.view());
}

#[test]
fn predict_lda_functional() {
    let ns = 10;
    let n = 1000;
    let nc = 16;
    let p = 4;

    let (noise, y) = gen_problem(n, ns, nc, -10, 10);
    let mut signal = Array2::<i16>::zeros((n, ns));
    signal
        .outer_iter_mut()
        .zip(y.iter())
        .for_each(|(mut s, y)| s.fill((y * 100) as i16));
    let x = &noise + signal;

    let lda = lda::LdaAcc::new(nc.into(), x.view(), y.view(), 0)
        .lda(p)
        .unwrap();
    let prs = lda.predict_proba(x.view());

    prs.outer_iter().zip(y.iter()).for_each(|(prs, y)| {
        let yguess = prs.argmax().unwrap();
        assert_eq!(yguess, *y as usize);
    });
}
