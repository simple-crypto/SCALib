use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::snr;

fn bench_get_snr(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_snr");
    let nc = 64;
    let np = 16;
    let n = 1000;
    for ns in [1000, 5000, 10000, 20000].iter() {
        let mut snr = snr::SNR::new(nc, *ns, np);
        let x = Array2::<i16>::random((n, *ns), Uniform::new(0, 10000));
        let y = Array2::<u16>::random((n, np), Uniform::new(0, nc as u16));
        snr.update(x.view(), y.view());

        group.bench_with_input(BenchmarkId::new("get_snr", ns), ns, |b, ns| {
            b.iter(|| {
                let res = snr.get_snr();
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(50);
    targets = bench_get_snr
}
criterion_main!(benches);
