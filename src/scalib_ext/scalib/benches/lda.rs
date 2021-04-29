use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::lda;

extern crate openblas_src as _;

fn bench_lda(c: &mut Criterion) {
    let mut group = c.benchmark_group("LDA");
    let nc = 64;
    let p = 20;
    let n = 10000;
    for i in [100, 200, 400, 600, 800, 1000, 1400, 1600, 1800, 2000].iter() {
        group.bench_with_input(BenchmarkId::new("LapackGeigen", i), i, |b, i| {
            b.iter(|| {
                let mut lda = lda::LDA::new(nc, p, *i);
                let x = Array2::<i16>::random((n, *i), Uniform::new(0, 10000));
                let y = Array1::<u16>::random(n, Uniform::new(0, nc as u16));
                lda.fit(x.view(), y.view(), 0);
            })
        });
        group.bench_with_input(BenchmarkId::new("GEigenSolver", i), i, |b, i| {
            b.iter(|| {
                let mut lda = lda::LDA::new(nc, p, *i);
                let x = Array2::<i16>::random((n, *i), Uniform::new(0, 10000));
                let y = Array1::<u16>::random(n, Uniform::new(0, nc as u16));
                lda.fit(x.view(), y.view(), 1);
            })
        });

        group.bench_with_input(BenchmarkId::new("GEigenSolverP", i), i, |b, i| {
            b.iter(|| {
                let mut lda = lda::LDA::new(nc, p, *i);
                let x = Array2::<i16>::random((n, *i), Uniform::new(0, 10000));
                let y = Array1::<u16>::random(n, Uniform::new(0, nc as u16));
                lda.fit(x.view(), y.view(), 2);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lda);
criterion_main!(benches);
