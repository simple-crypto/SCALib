use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;
use std::time::Duration;

fn bench_mttest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttest_update");
    let n = 5000;
    for d in [2, 3].iter() {
        for traces_len in [5000, 10000, 20000].iter() {
            let traces = Array2::<i16>::random((n, *traces_len), Uniform::new(0, 1000));
            let y = Array1::<u16>::random((n,), Uniform::new(0, 2));

            let mut tt = ttest::Ttest::new(*traces_len, *d as usize);
            tt.update(traces.view(), y.view());
            group.bench_with_input(
                BenchmarkId::new(format!("ttest_{}", traces_len), *d),
                d,
                |b, d| {
                    b.iter(|| {
                        tt.update(traces.view(), y.view());
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.01).sample_size(500).measurement_time(Duration::from_secs(30));
    targets = bench_mttest
}
criterion_main!(benches);
