use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;

fn bench_mttest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttest_update");
    let d = 2;
    let traces_len = 5000;
    let n = 100;
    let traces = Array2::<i16>::random((n, traces_len), Uniform::new(0, 10000));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));

    for npois in [1000, 5000, 10000, 20000, 50000, 100000].iter() {
        let pois = Array2::<u64>::random((d, *npois), Uniform::new(0,traces_len as u64));
        
        let mut mtt = ttest::MTtest::new(d,pois.view());
        mtt.update(traces.view(), y.view());
        group.bench_with_input(BenchmarkId::new("mttest", npois), npois, |b, npois| {
            b.iter(|| {
                mtt.update(traces.view(),y.view());
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(50);
    targets = bench_mttest
}
criterion_main!(benches);
