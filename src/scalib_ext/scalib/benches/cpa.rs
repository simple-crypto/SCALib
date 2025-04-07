//use std::fmt::format;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform};
use scalib::cpa::CPA;
use scalib::{AccType32bit, Config};

use rand_xoshiro::Xoshiro256StarStar as Prng;

type BenchMarkGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

// Generate the traces, label and model used for benchmarking
fn gen_cpa_inputs(
    seed: &u32,
    nc: u32,
    n: u32,
    nv: u32,
    ns: u32,
) -> (Array2<i16>, Array2<u16>, Array3<f64>) {
    // Randomness generator
    let seed = *seed as u64;
    let mut rng = Prng::seed_from_u64(seed);
    // Generate the inputs
    let models = Array3::<f64>::random_using(
        (nv as usize, nc as usize, ns as usize),
        Uniform::new(0.0, 1.0),
        &mut rng,
    );
    let traces =
        Array2::<i16>::random_using((n as usize, ns as usize), Uniform::new(0, 10), &mut rng);
    let labels = Array2::<u16>::random_using(
        (nv as usize, n as usize),
        Uniform::new(0, nc as u16),
        &mut rng,
    );
    (traces, labels, models)
}

fn bench_ll_cpa_all(seed: u32, nc: u32, n: u32, nv: u32, ns: u32, group: &mut BenchMarkGroup) {
    group.bench_with_input(
        BenchmarkId::new(format!("[ALL]-nc:{}-n:{}-nv:{}-ns:{}", nc, n, nv, ns), ns),
        &ns,
        |b, _| {
            // Create the CPA
            let config = Config::no_progress();
            // Create the input
            let (traces, labels, models) = gen_cpa_inputs(&seed, nc, n, nv, ns);
            let mut cpa = CPA::<AccType32bit>::new(nc as usize, ns as usize, nv as usize);
            b.iter(|| {
                let _ = cpa.update(traces.view(), labels.view(), &config);
                let _corr = cpa.compute_cpa(models.view());
            })
        },
    );
}

fn bench_ll_cpa_update(seed: u32, nc: u32, n: u32, nv: u32, ns: u32, group: &mut BenchMarkGroup) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("[UPDATE]-nc:{}-n:{}-nv:{}-ns:{}", nc, n, nv, ns),
            ns,
        ),
        &ns,
        |b, _| {
            // Create the CPA
            let config = Config::no_progress();
            let mut cpa = CPA::<AccType32bit>::new(nc as usize, ns as usize, nv as usize);
            // Create the input
            let (traces, labels, _models) = gen_cpa_inputs(&seed, nc, n, nv, ns);
            b.iter(|| {
                let _ = cpa.update(traces.view(), labels.view(), &config);
            })
        },
    );
}

fn bench_ll_cpa_correlation(
    seed: u32,
    nc: u32,
    n: u32,
    nv: u32,
    ns: u32,
    group: &mut BenchMarkGroup,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("[CORRELATION]-nc:{}-n:{}-nv:{}-ns:{}", nc, n, nv, ns),
            ns,
        ),
        &ns,
        |b, _| {
            // Create the CPA
            let config = Config::no_progress();
            let mut cpa = CPA::<AccType32bit>::new(nc as usize, ns as usize, nv as usize);
            // Create the input
            let (traces, labels, models) = gen_cpa_inputs(&seed, nc, n, nv, ns);
            let _ = cpa.update(traces.view(), labels.view(), &config);
            b.iter(|| {
                let _corr = cpa.compute_cpa(models.view());
            })
        },
    );
}

fn bench_ll_cpa(seed: u32, nc: u32, n: u32, nv: u32, ns: u32, group: &mut BenchMarkGroup) {
    bench_ll_cpa_all(seed, nc, n, nv, ns, group);
    bench_ll_cpa_update(seed, nc, n, nv, ns, group);
    bench_ll_cpa_correlation(seed, nc, n, nv, ns, group);
    println!("");
}

fn bench_cpa(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPA");
    let seed = 1;
    // nc, n, nv, ns
    bench_ll_cpa(seed, 256, 100000, 1, 1, &mut group);
    bench_ll_cpa(seed, 256, 100000, 1, 256, &mut group);
    bench_ll_cpa(seed, 256, 100000, 1, 2048, &mut group);
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_cpa
}
criterion_main!(benches);
