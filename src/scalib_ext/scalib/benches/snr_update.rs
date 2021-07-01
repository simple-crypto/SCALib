use criterion::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::snr;

fn bench_snr_update(c: &mut Criterion) {
    //let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("snr_update");
    //group.plot_config(plot_config);

    let nc = 256;
    let np = 4;
    let n = nc * 100;
    for ns in (10..21).map(|x| 2.0_f64.powi(x) as usize) {
        let x = Array2::<i16>::random((n, ns), Uniform::new(0, 10000));
        let y = Array2::<u16>::random((n, np), Uniform::new(0, nc as u16));

        let mut snr = snr::SNR::new(nc, ns, np);
        snr.update(x.view(), y.view(), 1 << 10);

        group.bench_with_input(BenchmarkId::new("chunks_25", ns), &ns, |b, ns| {
            b.iter(|| {
                snr.update(x.view(), y.view(), 1 << 25);
            })
        });
        group.bench_with_input(BenchmarkId::new("chunks_8", ns), &ns, |b, ns| {
            b.iter(|| {
                snr.update(x.view(), y.view(), 1 << 8);
            })
        });
        group.bench_with_input(BenchmarkId::new("chunks_10", ns), &ns, |b, ns| {
            b.iter(|| {
                snr.update(x.view(), y.view(), 1 << 10);
            })
        });
        group.bench_with_input(BenchmarkId::new("chunks_12", ns), &ns, |b, ns| {
            b.iter(|| {
                snr.update(x.view(), y.view(), 1 << 12);
            })
        });
        group.bench_with_input(BenchmarkId::new("chunks_14", ns), &ns, |b, ns| {
            b.iter(|| {
                snr.update(x.view(), y.view(), 1 << 14);
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default();
    targets = bench_snr_update
}
criterion_main!(benches);
