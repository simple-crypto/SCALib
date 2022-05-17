use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::snr;

fn bench_get_snr(c: &mut Criterion) {
    let nc = 256;
    let np = 16;
    let n = 1000;
    for i in [32, 64] {
        let mut group = c.benchmark_group(format!("get_snr_{}", i));
        for ns in [1000, 10000, 100000].iter() {
            let x = Array2::<i16>::random((n, *ns), Uniform::new(0, 10000));
            let y = Array2::<u16>::random((np, n), Uniform::new(0, nc as u16));
            if i == 32 {
                let mut snr = snr::SNR::<snr::SnrType32bit>::new(nc, *ns, np);
                snr.update(x.view(), y.view());
                group.bench_with_input(BenchmarkId::new("get_snr", ns), ns, |b, ns| {
                    b.iter(|| {
                        let res = snr.get_snr();
                    })
                });
            }
            if i == 64 {
                let mut snr = snr::SNR::<snr::SnrType64bit>::new(nc, *ns, np);
                snr.update(x.view(), y.view());
                group.bench_with_input(BenchmarkId::new("get_snr", ns), ns, |b, ns| {
                    b.iter(|| {
                        let res = snr.get_snr();
                    })
                });
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(50);
    targets = bench_get_snr
}
criterion_main!(benches);
