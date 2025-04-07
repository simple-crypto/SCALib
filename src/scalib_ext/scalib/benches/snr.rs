use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::{lvar, snr, Config};

type BenchMarkGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn gen_traces(n: usize, ns: usize) -> Array2<i16> {
    Array2::<i16>::random((n, ns), Uniform::new(0, 100))
}
fn gen_classes(np: usize, n: usize, nc: usize) -> Array2<u16> {
    Array2::<u16>::random((np, n), Uniform::new(0, nc as u16))
}

fn bench_get_snr_inner<S: lvar::AccType>(
    nc: usize,
    np: usize,
    ns: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    group.bench_with_input(BenchmarkId::new("get_snr", ns), &ns, |b, _| {
        let x = gen_traces(n, ns);
        let y = gen_classes(np, n, nc);
        let mut snr = snr::SNR::<S>::new(nc, ns, np);
        snr.update(x.view(), y.view(), &Config::no_progress())
            .unwrap();
        b.iter(|| {
            snr.get_snr();
        })
    });
}

fn bench_snr_update_inner<S: lvar::AccType>(
    nc: usize,
    np: usize,
    ns: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    group.bench_with_input(
        BenchmarkId::new(format!("chunk_{}", np), ns),
        &ns,
        |b, _| {
            let x = gen_traces(n, ns);
            let y = gen_classes(np, n, nc);
            let mut snr = snr::SNR::<S>::new(nc, ns, np);
            b.iter(|| {
                snr.update(x.view(), y.view(), &Config::no_progress())
                    .unwrap();
            })
        },
    );
}

fn bench_get_snr(c: &mut Criterion) {
    let nc = 256;
    let np = 16;
    let n = 1000;
    for i in [32, 64] {
        let mut group = c.benchmark_group(format!("get_snr_{}", i));
        for ns in [1000, 100000] {
            if i == 32 {
                bench_get_snr_inner::<lvar::AccType32bit>(nc, np, ns, n, &mut group);
            }
            if i == 64 {
                bench_get_snr_inner::<lvar::AccType64bit>(nc, np, ns, n, &mut group);
            }
        }
        group.finish();
    }
}

fn bench_snr_update(c: &mut Criterion) {
    let nc = 256;
    let n = 10000;

    for i in [32, 64] {
        let mut group = c.benchmark_group(format!("snr_update_{}", i));
        for ns in [1 << 10, 1 << 16] {
            for np in [1, 16] {
                if i == 32 {
                    bench_snr_update_inner::<lvar::AccType32bit>(nc, np, ns, n, &mut group);
                }
                if i == 64 {
                    bench_snr_update_inner::<lvar::AccType64bit>(nc, np, ns, n, &mut group);
                }
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_get_snr, bench_snr_update
}
criterion_main!(benches);
