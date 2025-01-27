use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scalib::multi_lda::{Class, MultiLdaAcc};

use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;

type BenchMarkGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn gen_traces(n: usize, ns: usize) -> Array2<i16> {
    Array2::<i16>::random((n, ns), Uniform::new(0, 100))
}
fn gen_classes(np: usize, n: usize, nc: usize) -> Array2<u16> {
    Array2::<u16>::random((np, n), Uniform::new(0, nc as u16))
}

/*
fn bench_snr_update_inner<S: snr::SnrType<Sample = i16>>(
    nc: usize,
    np: usize,
    ns: usize,
    n: usize,
    group: &mut BenchMarkGroup,
) {
    let x = gen_traces(n, ns);
    let y = gen_classes(np, n, nc);
    let mut snr = snr::SNR::<S>::new(nc, ns, np);
    group.bench_with_input(
        BenchmarkId::new(format!("chunk_{}", np), ns),
        &ns,
        |b, _| {
            b.iter(|| {
                snr.update(x.view(), y.view(), &Config::no_progress())
                    .unwrap();
            })
        },
    );
}

fn bench_snr_update(c: &mut Criterion) {
    let nc = 256;
    let n = 10000;

    for i in [32, 64] {
        let mut group = c.benchmark_group(format!("snr_update_{}", i));
        for ns in [1 << 10, 1 << 16] {
            for np in [1, 16] {
                if i == 32 {
                    bench_snr_update_inner::<snr::SnrType32bit>(nc, np, ns, n, &mut group);
                }
                if i == 64 {
                    bench_snr_update_inner::<snr::SnrType64bit>(nc, np, ns, n, &mut group);
                }
            }
        }
        group.finish();
    }
}
*/

fn gen_rnd_pois(npois: usize, nv: usize, ns: u32) -> Vec<Vec<u32>> {
    let mut test: Vec<Vec<u32>> = Vec::new();
    let mut rng = thread_rng();
    let mut tmpvec: Vec<u32> = (0..ns).collect();
    for _i in 0..nv {
        tmpvec.shuffle(&mut rng);
        test.push(tmpvec[0..npois].to_vec());
    }
    test
}

fn bench_mlda_update_inner(
    nc: Class,
    ns: u32,
    nv: usize,
    n: usize,
    npois: usize,
    group: &mut BenchMarkGroup,
) {
    // Create the benh data (traces and classes)
    let t = gen_traces(n, ns as usize);
    let x = gen_classes(nv, n, nc as usize);
    let pois: Vec<Vec<u32>> = gen_rnd_pois(npois, nv, ns);
    // Instanciate the MLDAAc
    let mut mlda = MultiLdaAcc::new(ns, nc, pois).unwrap();
    // Generate the MLDA
    group.bench_with_input(BenchmarkId::new(format!("test-{}", ns), ns), &ns, |b, _| {
        b.iter(|| {
            mlda.update(t.view(), x.view()).unwrap();
        })
    });
}

fn bench_mlda_update(c: &mut Criterion) {
    let nc = 256;
    let n = 10000;
    let npois: usize = 128;
    // Benchmark group
    for _nv in [16, 512, 1024] {
        let mut group = c.benchmark_group(format!("test MLDA update np:{}", _nv));
        for _ns in [1000, 10000] {
            bench_mlda_update_inner(nc, _ns, _nv, n, npois, &mut group);
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_mlda_update
}
criterion_main!(benches);
