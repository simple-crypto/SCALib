use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::mttest;
use std::time::Duration;

fn bench_mttest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttest_update");
    let d = 2;
    let traces_len = 1000;
    let n = 2048;
    let traces = Array2::<i16>::random((n, traces_len), Uniform::new(0, 1000));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));
    for d in [2].iter() {
        for npois in [100000].iter() {
            let pois = Array2::<u32>::random((*d, *npois), Uniform::new(0, traces_len as u32));

            let mut mtt = mttest::MTtest::new(*d, pois.view());
            mtt.update(traces.view(), y.view());
            group.bench_with_input(
                BenchmarkId::new(format!("mttest_{}", *d), npois),
                npois,
                |b, npois| {
                    b.iter(|| {
                        mtt.update(traces.view(), y.view());
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
    config = Criterion::default().significance_level(0.01).sample_size(500).measurement_time(Duration::from_secs(60));
    targets = bench_mttest
}
criterion_main!(benches);
