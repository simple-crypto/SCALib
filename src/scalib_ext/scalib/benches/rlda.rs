//use std::fmt::format;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform};
use scalib::rlda::RLDA;

use ndarray_rand::rand::prelude::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar;

type BenchMarkGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn gen_traces(n: usize, ns: usize) -> Array2<i16> {
    Array2::<i16>::random((n, ns), Uniform::new(0, 100))
}
fn gen_classes(nv: usize, n: usize, nb: u64) -> Array2<u64> {
    Array2::<u64>::random((n, nv), Uniform::new(0, (1 << nb) as u64))
}

fn generate_case_data(nv: usize, nb: u64, ns: u32, n: usize) -> (Array2<i16>, Array2<u64>) {
    // Genereate the useful data
    let t = gen_traces(n, ns as usize);
    let x = gen_classes(nv, n, nb);
    (t, x)
}

fn bench_rlda(
    seed: u32,
    ns: u32,
    nb: u32,
    n: u32,
    nv: u32,
    p: u32,
    ntest: u32,
    group: &mut BenchMarkGroup,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "RLDA-UNI-TEST ns:{} ; nb:{} ; n:{} ; nv:{} ; p:{} ; ntest:{}",
                ns, nb, n, nv, p, ntest
            ),
            nv,
        ),
        &nv,
        |b, _| {
            // RNG
            let seed = seed as u64;
            let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

            // Generate inputs
            let traces = Array2::<i16>::random_using(
                (n as usize, ns as usize),
                Uniform::new(0, 10),
                &mut rng,
            );

            let labels = Array2::<u64>::random_using(
                (nv as usize, n as usize),
                Uniform::new(0, (1 << nb) as u64),
                &mut rng,
            );

            // Create the CPA
            let mut rlda = RLDA::new(nb as usize, ns as usize, nv as usize, p as usize);

            // Fit traces
            rlda.update(traces.view(), labels.view(), 1);

            // Solve
            let _ = rlda.solve();

            // Test data
            let test_traces = Array2::<i16>::random_using(
                (ntest as usize, ns as usize),
                Uniform::new(0, 10),
                &mut rng,
            );

            let test_labels = Array2::<u64>::random_using(
                (nv as usize, ntest as usize),
                Uniform::new(0, (1 << nb) as u64),
                &mut rng,
            );

            ////////
            b.iter(|| {
                let _l2p1s = rlda.predict_log2p1(test_traces.view(), test_labels.view());
            })
        },
    );
}

fn bench_basic(c: &mut Criterion) {
    let mut group = c.benchmark_group("RLDA");
    // seed, ns, nb, n, nv, p, ntest
    bench_rlda(0, 1, 28, 10000, 1, 1, 1, &mut group);
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_basic
}
criterion_main!(benches);
