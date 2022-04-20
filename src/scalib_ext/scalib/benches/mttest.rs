use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::ttest;
use std::time::Duration;

fn bench_mttest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttest_update");
    let d = 2;
    let traces_len = 50000;
    let n = 10;
    let traces = Array2::<i16>::random((n, traces_len), Uniform::new(0, 1000));
    let y = Array1::<u16>::random((n,), Uniform::new(0, 2));
    let csize = 1 << 12;
    for d in [3].iter() {
        for npois in [50000, 100000, 200000, 500000 ].iter() {
            for csize in [*npois, 1<<4, 1<<6, 1<<8,1<<10, 1<<12].iter(){
                let pois = Array2::<u64>::random((*d, *npois), Uniform::new(0, traces_len as u64));

                let mut mtt = ttest::MTtest::new(*d, pois.view());
                mtt.update(traces.view(), y.view(), *csize);
                group.bench_with_input(
                    BenchmarkId::new(format!("mttest_{}_{}", *d, *csize), npois),
                    npois,
                    |b, npois| {
                        b.iter(|| {
                            mtt.update(traces.view(), y.view(), *csize);
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(100).measurement_time(Duration::from_secs(10));
    targets = bench_mttest
}
criterion_main!(benches);
