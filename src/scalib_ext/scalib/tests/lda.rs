use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use scalib::lda;

extern crate openblas_src as _;
#[test]
fn fit_lda() {
    let ns = 100;
    let n = 1000;
    let nc = 16;
    let p = 4;

    let mut lda = lda::LDA::new(nc, p, ns);
    let x = Array2::<i16>::random((n, ns), Uniform::new(0, 100));
    let y = Array1::<u16>::random(n, Uniform::new(0, nc as u16));

    lda.fit(x.view(), y.view(), 0);
}

#[test]
fn predict_lda() {
    let ns = 100;
    let n = 1000;
    let nc = 16;
    let p = 4;

    let mut lda = lda::LDA::new(nc, p, ns);
    let x = Array2::<i16>::random((n, ns), Uniform::new(0, 100));
    let y = Array1::<u16>::random(n, Uniform::new(0, nc as u16));

    lda.fit(x.view(), y.view(), 0);

    lda.predict_proba(x.view());
}

#[test]
fn predict_lda_functional() {
    let ns = 10;
    let n = 1000;
    let nc = 16;
    let p = 4;

    let mut lda = lda::LDA::new(nc, p, ns);
    let noise = Array2::<i16>::random((n, ns), Uniform::new(-10, 10));
    let mut signal = Array2::<i16>::zeros((n, ns));
    let y = Array1::<u16>::random(n, Uniform::new(0, nc as u16));
    signal
        .outer_iter_mut()
        .zip(y.iter())
        .for_each(|(mut s, y)| s.fill((y * 100) as i16));
    let x = &noise + signal;

    lda.fit(x.view(), y.view(), 0);
    let prs = lda.predict_proba(x.view());

    prs.outer_iter().zip(y.iter()).for_each(|(prs, y)| {
        let yguess = prs.argmax().unwrap();
        assert_eq!(yguess, *y as usize);
    });
}
