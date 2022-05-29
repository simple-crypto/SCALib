use criterion::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::snr;

fn gen_traces(n: usize, ns: usize) -> Array2<i16> {
    Array2::<i16>::random((n, ns), Uniform::new(0, 10000))
}
fn gen_classes(np: usize, n: usize, nc: usize) -> Array2<u16> {
    Array2::<u16>::random((np, n), Uniform::new(0, nc as u16))
}

fn bench_snr_update(c: &mut Criterion) {
    //let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    //group.plot_config(plot_config);

    let nc = 256;
    let n = 10000;

    for i in [32, 64] {
        let mut group = c.benchmark_group(format!("snr_update_{}", i));
        for ns in [10, 16].iter().map(|x| 2.0_f64.powi(*x) as usize) {
            let x = gen_traces(n, ns);
            //for np in [1, 4, 8, 16].iter() {
            for np in [1, 16].iter() {
                let y = gen_classes(*np, n, nc);
                if i == 32 {
                    let mut snr = snr::SNR::<snr::SnrType32bit>::new(nc, ns, *np);
                    group.bench_with_input(
                        BenchmarkId::new(format!("chunk_{}", *np), ns),
                        &ns,
                        |b, ns| {
                            b.iter(|| {
                                snr.update(x.view(), y.view());
                            })
                        },
                    );
                }
                if i == 64 {
                    let mut snr = snr::SNR::<snr::SnrType64bit>::new(nc, ns, *np);
                    group.bench_with_input(
                        BenchmarkId::new(format!("chunk_{}", *np), ns),
                        &ns,
                        |b, ns| {
                            b.iter(|| {
                                snr.update(x.view(), y.view());
                            })
                        },
                    );
                }
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().sample_size(20);
    targets = bench_snr_update
}
criterion_main!(benches);
