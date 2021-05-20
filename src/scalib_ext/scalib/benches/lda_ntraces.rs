use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::lda;

extern crate openblas_src as _;

fn bench_lda(c: &mut Criterion) {
    let mut group = c.benchmark_group("LDA_ntraces");
    let nc = 64;
    let p = 20;
    let ns = 100;
    let n = 10000;
    for i in [50000, 70000, 100000, 150000, 200000, 250000].iter() {
        group.bench_with_input(BenchmarkId::new("LapackGeigen", i), i, |b, i| {
            b.iter(|| {
                let mut lda = lda::LDA::new(nc, p, ns);
                let x = Array2::<i16>::random((*i, ns), Uniform::new(0, 10000));
                let y = Array1::<u16>::random(*i, Uniform::new(0, nc as u16));
                lda.fit(x.view(), y.view(), 0);
            })
        });
        group.bench_with_input(BenchmarkId::new("GEigenSolverP", i), i, |b, i| {
            b.iter(|| {
                let mut lda = lda::LDA::new(nc, p, ns);
                let x = Array2::<i16>::random((*i, ns), Uniform::new(0, 10000));
                let y = Array1::<u16>::random(*i, Uniform::new(0, nc as u16));
                lda.fit(x.view(), y.view(), 2);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lda);
criterion_main!(benches);
